# Project Overview: Driver Monitoring System

Documento tecnico de referencia: o que o repositorio faz, como faz e como as tecnologias novas foram integradas na modernizacao de outubro de 2026. Para o plano de evolucao, veja [`MASTER_EVOLUTION_PLAN.md`](MASTER_EVOLUTION_PLAN.md); para fases e ferramentas, [`ROADMAP_AND_SKILLS.md`](ROADMAP_AND_SKILLS.md).

## 1. O que o sistema faz

O DMS (Driver Monitoring System) observa o rosto do motorista por camera e transforma sinais visuais em um **Attention Score de 0 a 100**, com rotulo (`Atento`, `Alerta`, `Distraido`) e eventos explicaveis:

| Evento | Sinal | Regra padrao (`shared/scoring-config.json`) |
|---|---|---|
| Sonolencia | olhos fechados (EAR < 0.21) | fechamento continuo >= 1.2 s |
| Microssono | olhos fechados | fechamento continuo >= 2.0 s |
| Fadiga | PERCLOS (fracao do tempo com olhos fechados) | >= 15% na janela de 60 s, com no minimo 10 s observados |
| Bocejo | Mouth Aspect Ratio (MAR) | MAR >= 0.6 por >= 1.5 s |
| Olhar fora | pose da cabeca (yaw > 20 graus ou pitch > 15 graus) | desvio continuo >= 1.5 s |
| Distracao acumulada | pose da cabeca, acumulada | >= 10 s fora da via dentro de 30 s (Euro NCAP 2026 VATS) |
| Celular / Celular suspeito | YOLO (Python) ou MediaPipe Object Detector (web) + filtro espacial | celular perto do rosto / no quadro |
| Sem rosto | ausencia de landmarks | > 1.0 s sem rosto |

Cada evento ativo aplica uma penalidade; sem penalidades ha um bonus de estabilidade. O score bruto e suavizado por media movel exponencial (`score_smoothing = 0.8`) para evitar oscilacao.

Ha duas superficies:

- **App web** (portfolio, Vercel): webcam no navegador com MediaPipe Face Landmarker, demo roteirizada (`Demo` e `Review`), dashboard de telemetria, exportacao da sessao, PWA.
- **Engine Python** (local): camera ou video com OpenCV, MediaPipe Tasks (rosto e maos), Ultralytics YOLO para celular, overlay OpenCV, modo headless.

## 2. Arquitetura

```text
                    shared/scoring-config.json  (fonte unica das regras)
                               |
            +------------------+-------------------+
            |                                      |
   Python (referencia)                       TypeScript (web)
   dms/ear.py, mouth.py, rolling.py,         src/domain/eyeState.ts, mouthState.ts,
   attention.py                              rolling.ts, attention.ts
            |                                      |
   scripts/generate_scoring_scenarios.py  ---->  shared/scoring-scenarios.json
            |                                      |
   tests/test_scoring_parity.py             tests/web/scoringParity.test.ts
```

### 2.1 App web (`src/`)

Camadas com dependencias apontando para dentro (Clean Architecture leve, sem framework):

| Camada | Responsabilidade | Arquivos |
|---|---|---|
| `domain/` | Regras puras e testaveis: EAR, MAR, PERCLOS/VATS, score, sessao, relatorio, sensibilidade | `attention.ts`, `eyeState.ts`, `mouthState.ts`, `rolling.ts`, `session.ts`, `report.ts`, `config.ts` |
| `app/` | Casos de uso: pipeline por frame, sinais da demo, controller do monitor | `pipeline.ts`, `signals.ts`, `monitorController.ts` |
| `vision/` | MediaPipe (runtime WASM compartilhado), Face Landmarker, detector de celular, tracking com EMA e hold/fade, mapeamento video->canvas, overlay | `visionRuntime.ts`, `faceLandmarker.ts`, `phoneDetector.ts`, `faceTracker.ts`, `coordinateMapper.ts`, `overlayRenderer.ts`, `headPose.ts` |
| `infra/` | Camera, analytics opt-in, service worker, download local | `camera.ts`, `analytics.ts`, `pwa.ts`, `download.ts` |
| `ui/` | Unico ponto que escreve no DOM | `monitorView.ts`, `dom.ts` |
| `main.ts` | Composition root: monta o grafo de objetos e injeta dependencias | |

Fluxo de um frame no modo Camera:

1. `requestAnimationFrame` chama `MonitorController.frame`.
2. `FaceLandmarkerService.detect` roda inferencia em modo VIDEO, limitada a 30 Hz.
3. `CoordinateMapper` converte landmarks normalizados para pixels do canvas, considerando `object-fit: cover`, `devicePixelRatio` e espelhamento.
4. `FaceTracker` suaviza os pontos (EMA adaptativa), calcula EAR, MAR e pose, e mantem o overlay por alguns frames quando o rosto some (hold e fade).
   Em paralelo, `PhoneDetectorService` roda o Object Detector no maximo a cada 400 ms; as caixas vao para o mesmo espaco do canvas e `isPhoneNearFace` (porta de `dms/spatial.py`) decide entre "Celular" e "Celular suspeito".
5. `MonitoringPipeline` passa os sinais por `EyeStateTracker`, `MouthStateTracker` e `AttentionScorer`, e agrega a sessao em `SessionStats` (eventos contados por ocorrencia, nao por frame).
6. `MonitorView` atualiza HUD a cada frame e dashboard a cada 250 ms.

No modo Demo/Review, `demoSignals` gera sinais deterministicos a 30 Hz fixos que passam pelo **mesmo** pipeline, entao a demo exercita as regras reais.

### 2.2 Engine Python (`dms/`, `main.py`)

`DMSPipeline` orquestra colaboradores atras de `Protocol`s (`FaceSource`, `HeadPoseSource`, `PhoneDetector`, `HandSource`...). `main.py` e o composition root: escolhe implementacoes, degrada com elegancia quando YOLO/torch nao estao disponiveis e trata erros de dominio (`DMSError`) uma vez, na borda da CLI, com codigos de saida estaveis (2 config, 3 fonte de video, 4 download de modelo).

Pontos tecnicos:

- **Relogio por frame (`FrameClock`)**: tempo do video para arquivos, relogio monotonic para camera; garante timestamps estritamente crescentes (exigencia do MediaPipe VIDEO) e regras de duracao corretas mesmo processando mais rapido que o tempo real.
- **Detectores em passo (`detector_interval_frames`)**: YOLO e maos rodam a cada N frames (padrao 3) e o resultado e reutilizado entre execucoes. Medido em CPU a 720p: **16.8 -> 40.6 FPS**.
- **Download de modelos seguro**: HTTPS com allowlist de host, escrita atomica via arquivo temporario.

## 3. Logica central

### 3.1 EAR (Eye Aspect Ratio)

Soukupova e Cech (2016): `EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)`, media dos dois olhos. Um fechamento curto (2 a 6 frames) conta como piscada; so fechamento sustentado vira sonolencia ou microssono.

### 3.2 MAR (Mouth Aspect Ratio)

Media de tres aberturas internas dos labios (pares 81-178, 13-14, 311-402) dividida pela largura entre os cantos internos (78-308). O bocejo exige abertura sustentada, o que separa bocejo de fala (cenario `talking_is_not_yawning`).

### 3.3 Janela deslizante ponderada por tempo (`RollingDutyCycle`)

Base de PERCLOS e VATS. Cada amostra guarda a contribuicao do intervalo que termina nela; somas sao mantidas incrementalmente (O(1) amortizado por frame). Intervalos longos (rosto perdido, frames descartados) contam no maximo `rolling_max_gap_s = 0.5 s`, para uma lacuna nao inflar a metrica. A implementacao e identica, operacao por operacao, em Python e TypeScript, o que permite comparar os resultados com 6 casas decimais.

### 3.4 Paridade entre runtimes

O motor Python e a referencia. `scripts/generate_scoring_scenarios.py` executa 14 cenarios deterministicos (piscada, sonolencia, microssono, olhar rapido, olhar sustentado, olhar para baixo, celular perto/longe, perda de rosto, PERCLOS alto, bocejo, fala, olhares repetidos) e grava as saidas esperadas. Os testes Python e Vitest validam o mesmo arquivo; o CI regenera e falha se houver diferenca. O mesmo script gera `shared/spatial-cases.json` (127 posicoes de celular combinadas com inclinacoes de cabeca) para a regra de proximidade celular-rosto, validada da mesma forma.

## 4. Qualidade, seguranca e privacidade

- **Tipagem estrita**: TypeScript `strict` + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`; ESLint `strictTypeChecked`; configuracao externa validada em `parseScoringConfig` e `load_shared_scoring`.
- **Erros**: `AppError` com codigos estaveis no web (camera negada, modelo nao carregou...) e fallback automatico para a demo; `DMSError` com exit codes no Python.
- **Headers**: CSP restritiva, `X-Frame-Options: DENY`, HSTS, `Permissions-Policy` (camera apenas na propria origem). O `vite preview` aplica os mesmos headers do `vercel.json`.
- **Privacidade**: video processado no dispositivo; relatorio gerado localmente; analytics opt-in. O coletor de estatisticas do `@mediapipe/tasks-vision` 1.1.x (envia contadores para `odml.pa.googleapis.com`) e bloqueado pela CSP de producao.
- **CSV seguro**: celulas iniciadas por `= + - @` sao neutralizadas contra injecao de formula em planilhas.

### Testes

| Camada | Ferramenta | Quantidade |
|---|---|---|
| Regras, sessao, relatorio, config, proximidade do celular (web) | Vitest | 45 |
| Engine, pipeline com fakes, CLI, overlay, janelas (Python) | unittest | 45 |
| Fluxos reais no build de producao | Playwright (Chromium, camera sintetica e foto real como camera) | 6 |

## 5. Como as tecnologias novas foram integradas

| Tecnologia | Versao | Integracao |
|---|---|---|
| `@mediapipe/tasks-vision` | 1.1.1 | Atualizada de 0.10.35; API compativel. Um teste garante que a versao do WASM no CDN e a mesma do bundle. Um unico fileset WASM serve o Face Landmarker e o Object Detector (EfficientDet-Lite0 float16) usado para celular. |
| `mediapipe` (Python) | 1.1.1 | `mp.solutions` foi removido no 1.x; o backend legado foi apagado e o Tasks API e o unico caminho. |
| `ultralytics` | 8.4.178 | Executado em passo de N frames. |
| OpenCV | 5.0.0.93 | `opencv-python` e `opencv-contrib-python` fixados na mesma versao (ambos instalam `cv2`). |
| Vite / Vitest / ESLint | 8.3.4 / 5.0.5 / 10.13.0 | Atualizacoes patch; `vite preview` com headers de producao. |
| Playwright | 1.64.1 | Nova suite E2E e job dedicado no CI. |
| TypeScript | 6.0.3 | 7.0.2 testado: compila, mas `typescript-eslint` 8.71.2 ainda nao suporta TS 7; mantido em 6. |

## 6. Executar

```bash
npm ci && npm run dev                 # web em http://localhost:3000
npm run validate                      # format, lint, typecheck, Vitest, build
npm run test:e2e                      # Playwright (rode antes: npx playwright install chromium)
py -3.12 -m venv .venv && .venv\Scripts\activate && pip install -r requirements-dev.txt
python main.py --source 0             # engine local
python -m unittest discover -s tests
```
