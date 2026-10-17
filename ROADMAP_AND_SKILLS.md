# Roadmap and Skills

Plano de acao da modernizacao: fases aplicadas, ferramentas e competencias usadas em cada uma, bibliotecas implementadas e bibliotecas gratuitas sugeridas para os proximos passos. Regra de ouro mantida: **somente solucoes open-source ou gratuitas**.

## 1. Fases aplicadas

### Fase 1: diagnostico, restauracao e continuidade

| Atividade | Resultado |
|---|---|
| Mapeamento de stack, docs e historico de commits | App web TypeScript + Vite e engine Python 3.12 ja em camadas na branch `refactor/phase3-architecture` |
| Validacao completa | `npm run validate`, `ruff`, `unittest`, regeneracao dos cenarios de paridade e smoke test headless do `main.py` passaram |
| Auditoria de dependencias | `npm audit` sem vulnerabilidades; desatualizadas: patches (Vite, Vitest, ESLint...), majors (`@mediapipe/tasks-vision` 1.x, TypeScript 7) |
| Achados | OpenCV duplicado no venv, gerador gravando CRLF no Windows, caminhos errados no README, backend `mp.solutions` morto |

**Skills e ferramentas:** leitura de codigo e de historico com `git log`, `npm outdated`, `npm audit`, `pip list --outdated`, `pip check`, video sintetico com OpenCV para smoke test.

### Fase 2: auditoria e refatoracao

| Mudanca | Por que |
|---|---|
| Patches de dependencias e OpenCV fixado em par | `mediapipe` exige `opencv-contrib-python` e `ultralytics` exige `opencv-python`; os dois instalam `cv2` |
| `@mediapipe/tasks-vision` 0.10.35 -> 1.1.1 | Versao atual; validada no navegador (478 landmarks) e por teste de sincronia JS/WASM |
| TypeScript 7 avaliado e revertido | `typescript-eslint` ainda nao suporta TS 7; lint estrito vale mais que o major |
| Remocao do backend `mp.solutions` | Inexistente no `mediapipe` 1.x; codigo morto com dois caminhos |
| `DMSConfig` le `shared/scoring-config.json` | Uma unica fonte de verdade para as regras nos dois runtimes |
| YOLO e maos a cada N frames | Gargalo de CPU; 16.8 -> 40.6 FPS a 720p |
| Gerador de cenarios grava LF | Arquivo identico em Windows e Linux (o CI faz diff) |

**Skills:** Clean Architecture, injecao de dependencias (Protocols em Python, interfaces em TS), tipagem estrita, profiling simples com `time.perf_counter`, engenharia reversa de bundle minificado (descoberta do coletor de estatisticas do MediaPipe).

### Fase 3: pesquisa de mercado e features

Pesquisa: protocolo Euro NCAP Safe Driving Driver Engagement v1.2 (2026), regulamentos UE 2023/2590 (ADDW) e 2021/1341 (DDAW), literatura de PERCLOS e MAR, repositorios open-source de DMS (openpilot, Driver-State-Detection, entre outros).

| Feature | Base | Status |
|---|---|---|
| PERCLOS (fadiga) | Wierwille et al. 1994; janela de 60 s | Implementado nos dois runtimes, limiar heuristico de 15% |
| Bocejo por MAR | Landmarks internos dos labios do Face Mesh | Implementado; limiar heuristico 0.6 por 1.5 s |
| Distracao acumulada (VATS) | Euro NCAP 2026: 10 s fora da via em 30 s | Implementado |
| Exportacao da sessao JSON/CSV | Necessidade de relatorio para frotas | Implementado, 100% local |
| E2E com camera sintetica | Fluxos reais com CSP de producao | Implementado e no CI |
| Celular no navegador | MediaPipe Object Detector + porta da regra espacial do Python | Implementado; paridade em 127 casos e E2E com foto real |

**Skills:** pesquisa regulatoria, modelagem de metricas temporais, design de cenarios deterministicos, UI de dashboard acessivel (contraste, foco visivel, `aria`), testes E2E.

### Fase 4: documentacao

`PROJECT_OVERVIEW.md`, este arquivo e `MASTER_EVOLUTION_PLAN.md`, alem da atualizacao do `README.md` (features, privacidade, testes, roadmap).

## 2. Bibliotecas implementadas

| Biblioteca | Versao | Licenca | Uso |
|---|---|---|---|
| `@mediapipe/tasks-vision` | 1.1.1 | Apache-2.0 | Face Landmarker no navegador |
| `mediapipe` (Python) | 1.1.1 | Apache-2.0 | Face e Hand Landmarker (Tasks) |
| `opencv-python` / `opencv-contrib-python` | 5.0.0.93 | Apache-2.0 | Captura, desenho e solvePnP |
| `numpy` | 2.5.4 | BSD-3 | Geometria dos landmarks |
| `ultralytics` | 8.4.178 | **AGPL-3.0** | YOLO para celular (somente engine local) |
| `torch` / `torchvision` | 2.14.1 / 0.29.1 | BSD-3 | Runtime do YOLO |
| EfficientDet-Lite0 float16 (modelo MediaPipe) | 1 | Licenca dos pesos nao informada na pagina do MediaPipe; verificar o model card antes de uso comercial | Celular no navegador |
| Vite / Vitest | 8.3.4 / 5.0.5 | MIT | Build e testes unitarios |
| Playwright | 1.64.1 | Apache-2.0 | E2E |
| ESLint + typescript-eslint, Prettier, Ruff | 10.13 / 8.71.2, 3.9.10, 0.17.1 | MIT | Qualidade |

Fixture de E2E: `tests/e2e/fixtures/driver-phone.jpg` (Wikimedia Commons, epSos.de, CC BY 2.0), creditada em `tests/e2e/fixtures/README.md`.

Observacao de licenca: o Ultralytics YOLO e AGPL-3.0. Como o projeto e open-source (MIT) e o YOLO roda somente localmente, isso e aceitavel, mas um produto fechado precisaria de licenca comercial ou de outro detector.

## 3. Bibliotecas e recursos gratuitos sugeridos

| Proximo passo | Recurso | Licenca / custo | Observacao |
|---|---|---|---|
| Celular com YOLO no navegador | ONNX Runtime Web (WebGPU/WASM) | MIT | Requer modelo exportado; atencao a licenca AGPL dos pesos Ultralytics |
| Olhar (gaze) | Landmarks de iris 468-477 do Face Landmarker | Apache-2.0 | Sem modelo extra |
| Olhar com modelo dedicado | L2CS-Net ou yakhyo/gaze-estimation (ONNX) | MIT | Evitar WebGazer (GPL-3.0, calibracao por clique) |
| Calibracao de limiares | YawDD e datasets publicos de sonolencia | Uso academico | Validar MAR e PERCLOS |
| Alerta sonoro | Web Audio API | Nativo | Faixa 200-8000 Hz alinhada ao ADDW |
| Confianca por landmark | Presence scores do tasks-vision 1.1 | Apache-2.0 | Descartar frames ruins antes do EAR |
| Telemetria opcional | Supabase/Neon no plano gratuito ou SQLite local | Gratuito | Somente com consentimento explicito |
| Acessibilidade automatizada | `@axe-core/playwright` | MPL-2.0 | Somar ao E2E existente |

## 4. Roadmap

| Horizonte | Item | Esforco |
|---|---|---|
| Curto | Celular no navegador: rodar em Web Worker e avaliar EfficientDet-Lite2 com GPU | M |
| Curto | Alerta sonoro opcional (Web Audio), com mute persistido | P |
| Curto | `axe-core` no E2E | P |
| Medio | Gaze por iris combinado com pose para "eyes off road" | M |
| Medio | Calibracao por motorista (EAR/MAR de base nos primeiros 30 s) | M |
| Medio | Relatorio de sessao tambem no engine Python (`--report out.json`), mesmo schema | P |
| Longo | Inferencia em Web Worker com `OffscreenCanvas` | G |
| Longo | Backend opcional de frotas com consentimento, ingestao do schema de relatorio | G |
