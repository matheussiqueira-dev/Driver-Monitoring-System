# Plano de Evolução — Driver-Monitoring-System

> Documento vivo, revisado semanalmente. Última revisão: 2026-10-23.
> Regra central: nenhuma mudança entra sem estar ligada a um objetivo desta página e sem manter ou melhorar as métricas da seção 2.

## 1. Identidade do projeto

- **Propósito:** demonstração técnica de monitoramento do motorista (sonolência, distração, celular) para avaliadores técnicos do portfólio, com interface web que roda o Face Landmarker no navegador e um engine Python/OpenCV de referência para câmera ou vídeo local.
- **Tipo:** pipeline de visão computacional (PWA estática + engine desktop Python).
- **Stack:** TypeScript 6.0.3 (strict) + Vite 8.3.4, `@mediapipe/tasks-vision` 1.1.1 (WASM por jsDelivr, modelos `face_landmarker/float16/1` e `efficientdet_lite0/float16/1` no Google Storage), Vitest 5.0.5, Playwright 1.64.1, ESLint 10.13.0 + typescript-eslint 8.71.2, Prettier 3.9.10; Node `24.x` (`engines`). Python 3.12 (`requires-python >=3.12,<3.13`) com `opencv-python`/`opencv-contrib-python` 5.0.0.93, `mediapipe` 1.1.1, `numpy` 2.5.4, `ultralytics` 8.4.178, `torch` 2.14.1, `torchvision` 0.29.1; `ruff` 0.17.1; testes com `unittest`.
- **Gerenciador de pacotes:** npm (`package-lock.json`); pip com `requirements.txt` / `requirements-dev.txt` (versões fixadas com `==`).
- **Deploy:** Vercel, saída estática (`vercel.json`: `framework: null`, `npm ci`, `npm run build`, `dist`, rewrites de `/monitor`, `/dashboard`, `/arquitetura`, `/produto`, `/sobre`). URL: https://driver-monitoring-system-two.vercel.app (campo `homepage`). O engine Python não vai para a Vercel (`.vercelignore`).
- **Estrutura:**
  - `src/domain/` regras puras (EAR, MAR, PERCLOS/VATS em `rolling.ts`, score, sessão, relatório, regra espacial celular-rosto).
  - `src/app/` casos de uso (pipeline, sinais roteirizados da demo, `monitorController.ts`).
  - `src/vision/` Face Landmarker, Object Detector, mapeamento de coordenadas e overlay em canvas.
  - `src/infra/` câmera, analytics, download do relatório, registro do service worker; `src/ui/` escrita no DOM.
  - `shared/` fonte única de limiares (`scoring-config.json`) e fixtures de paridade geradas pelo Python (`scoring-scenarios.json`, `spatial-cases.json`).
  - `dms/` + `main.py` engine Python (Face Mesh, solvePnP, YOLO, MediaPipe Hands, overlay OpenCV, CLI com `--headless`/`--max-frames`).
  - `tests/` (`unittest`), `tests/web/` (Vitest), `tests/e2e/` (Playwright com câmera sintética e foto real CC BY 2.0).
- **Comandos:** install `npm ci` · dev `npm run dev` · lint `npm run lint` + `npm run format:check` (web), `ruff check .` (Python) · typecheck `npm run typecheck` · test `npm test`, `npm run test:e2e`, `python -m unittest discover -s tests` · build `npm run build` · validação completa `npm run validate`.

## 2. Estado atual (baseline)

| Verificação | Comando | Resultado | Data |
| --- | --- | --- | --- |
| Instalação | `npm ci`; `uv venv --python 3.12` + subconjunto de `requirements-dev.txt` | PASS (web, 168 pacotes; avisos `EBADENGINE` com Node 22.22.0 local). Python: PASS parcial (numpy, opencv x2, mediapipe, ruff instalados; torch/torchvision/ultralytics só resolvidos em `--dry-run`, não instalados) | 2026-10-23 |
| Lint | `npm run format:check`; `npm run lint`; `ruff check .` | PASS / PASS / PASS. `ruff format --check .` (não exigido no CI): 4 arquivos fora do formato (`dms/overlay.py`, `dms/spatial.py`, `dms/spatial_cases.py`, bloco de código em `MASTER_EVOLUTION_PLAN.md`) | 2026-10-23 |
| Typecheck | `npm run typecheck` | PASS | 2026-10-23 |
| Testes | `npx vitest run`; `python -m unittest discover -s tests` | Vitest 45 passando / 0 falhando (8 arquivos); unittest 45 passando / 0 falhando. Paridade: `python scripts/generate_scoring_scenarios.py` sem diff em `shared/` | 2026-10-23 |
| Cobertura | — | não configurada (sem `@vitest/coverage-v8` nem `coverage.py`) | 2026-10-23 |
| Build | `npx vite build` | PASS; JS inicial 15,14 kB gzip, `vision_bundle` 45,64 kB gzip (sob demanda); `dist/` 6,6 MB, dos quais 6,0 MB são `assets/demo.gif` | 2026-10-23 |
| Vulnerabilidades (audit) | `npm audit --omit=dev --json`; `npm audit --json`; `pip-audit -r requirements.txt --no-deps` | 0 critical / 0 high / 0 moderate (prod e dev); pip-audit: nenhuma vulnerabilidade conhecida | 2026-10-23 |
| CI no GitHub | `gh run list --branch main` | `ci.yml` (jobs `web`, `e2e`, `python`); 2 últimas execuções na `main` com `success` (2026-10-17, `42f1550`) | 2026-10-23 |

Observações:
- E2E (`npm run test:e2e`) **não verificado** localmente: o download do Chromium do Playwright foi recusado pela rede do ambiente e não há navegador de sistema. O CI na `main` é a fonte de verdade (job `e2e` verde em 2026-10-17).
- Torch, torchvision e ultralytics **não instalados** (download de várias centenas de MB com dependências CUDA em máquina compartilhada); os testes Python não dependem deles porque `dms/detection.py` importa `ultralytics` de forma preguiçosa. Execução de `main.py` com câmera ou YOLO: **não verificada** (sem câmera e sem pesos).
- Domínio de produção: **não verificado** (requisição HTTP bloqueada pela rede do ambiente).
- Node local 22.22.0 diverge de `engines.node: 24.x`; o CI usa Node 24.

## 3. Histórico e trajetória

33 commits entre 2026-01-28 e 2026-10-17, em quatro fases:

1. **2026-01-28 a 2026-02-03 — protótipo Python.** Pipeline OpenCV + MediaPipe, fallback para Tasks API, seleção de câmera por nome, GIF de demonstração e licença MIT.
2. **2026-05-24 a 2026-05-30 — vitrine web.** Dashboard em JavaScript sem framework, deploy estático na Vercel (correção do preset Python), estabilização do overlay facial, analytics opt-in e `docs/PRODUCTION_AUDIT.md`.
3. **2026-10-11 — reprodutibilidade.** Dependências Python fixadas para 3.12, Node 24, modo `--headless`/`--max-frames` e erro explícito para fonte de vídeo indisponível.
4. **2026-10-15 a 2026-10-17 — modernização.** Reescrita para TypeScript + Vite em camadas (71 arquivos, +10,2 mil linhas), limiares unificados em `shared/scoring-config.json` com testes de paridade, PERCLOS/MAR/VATS nos dois runtimes, exportação JSON/CSV, CI com três jobs, headers de produção no `vite preview`, celular no navegador com Object Detector e correção de rotas limpas (PR #5).

Áreas de maior churn: `README.md` (14 alterações), `main.py` (11), `package.json` (8), `dms/config.py` (7), `index.html` e `vercel.json` (6 e 5). Padrão de commit: frases em inglês no imperativo, sem Conventional Commits; dois commits concentram mais de 22 mil linhas (difícil revisar e reverter). Regressão recorrente: rotas e preset de deploy na Vercel (2026-05-24 e 2026-10-17). Débitos herdados: a camada visual (Inter, paleta própria, textos sem acento, badge "LIVE") não acompanhou a reescrita do domínio.

## 4. Objetivos definidos (critério de parada)

| # | Eixo | Objetivo | Métrica | Baseline | Meta | Prioridade | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O1 | Funcionamento | Demo sem câmera honesta e rotulada | Ocorrências de "LIVE" visível fora do modo Câmera; e2e que exige o rótulo "Dados sintéticos" | "LIVE" sempre visível (`index.html:138-141`); 0 rótulos | 0 ocorrências; 1 teste e2e verde | P0 | ativo |
| O2 | Cobertura de testes | Medir e travar a cobertura nos dois runtimes | % de linhas (`vitest --coverage`, `coverage.py`) | não configurada | web ≥ 80% em `src/domain` e `src/app`; Python ≥ 75% em `dms/` sem `face_mesh`/`detection`; gate no CI | P1 | ativo |
| O3 | Segurança | Manter zero vulnerabilidades e fechar a CSP | `npm audit` high+critical; hosts externos em `connect-src`/`script-src` | 0; 2 hosts (jsDelivr, Google Storage) | 0; 0 hosts externos após self-host (se aprovado) | P2 | ativo |
| O4 | Performance | Reduzir o peso inicial da página | bytes transferidos no primeiro carregamento (mobile) | ~6,0 MB (GIF carregado por `<img>`) | ≤ 2,0 MB; Lighthouse mobile Performance ≥ 85 | P1 | ativo |
| O5 | Qualidade/Manutenibilidade | Formatação Python verificada no CI | arquivos fora de `ruff format --check` | 4 | 0, com passo no CI | P2 | ativo |
| O6 | Acessibilidade/UX | Interface em PT-BR correto e acessível | caracteres acentuados em `index.html`; violações axe serious/critical em 320/768/1280 | 0 acentos (texto sem acentuação); axe não verificado | textos com acento; 0 violações serious/critical | P1 | ativo |

## 5. Guardrails

- Mudança só com objetivo associado (O#) ou item de backlog (B#); refatoração sem ganho mensurável é proibida.
- Nenhuma métrica da seção 2 pode piorar; se piorar, a mudança é revertida.
- Upgrade de versão major, mudança de arquitetura, remoção de funcionalidade e alteração de deploy exigem aprovação do dono (entram em "Decisões pendentes").
- No máximo uma feature nova por semana, de risco baixo, atrás de testes.
- Limiares e penalidades só mudam em `shared/scoring-config.json`, seguidos de `python scripts/generate_scoring_scenarios.py` e dos testes de paridade dos dois lados.
- Não remover o engine Python nem `docs/PRODUCTION_AUDIT.md`; o Python é a implementação de referência.
- Nenhuma copy pode sugerir uso em veículo real; analytics nunca recebe landmarks, imagens ou métricas por quadro.
- `vercel.json` e `.vercelignore` só mudam com verificação em preview antes do merge.
- Pinagem de `torch`/`ultralytics` só muda junto com a versão instalada pelo CI.

## 6. Backlog priorizado

- P0 = quebra funcionamento/segurança · P1 = risco alto · P2 = qualidade · P3 = melhoria

- [ ] **B1 (P0, O1)** Rotular a demonstração e esconder "LIVE" fora do modo Câmera (`index.html:104-141`, `src/ui/monitorView.ts`) — e2e no modo Demo encontra "Dados sintéticos" e não encontra "LIVE"; no modo Câmera o badge aparece.
- [ ] **B2 (P0, O1)** Aviso "Demonstração técnica, não é dispositivo de segurança veicular" visível no topo do monitor — e2e encontra o texto em 375 px e 1440 px.
- [ ] **B3 (P1, O1)** Resolver o modo "Review" (`index.html:107`, `src/app/signals.ts:33`), hoje um roteiro sintético de maior risco com nome que sugere revisão da sessão — removido ou renomeado para "Cenário de risco (sintético)" conforme decisão do dono; e2e cobre o estado final.
- [ ] **B4 (P1, O1)** Aviso de privacidade antes de ligar a câmera ("o vídeo é processado no navegador e não sai do dispositivo; só métricas anônimas de desempenho") — e2e encontra o texto antes do clique em "Camera".
- [ ] **B5 (P1, O4)** Tirar o GIF de 6,0 MB do caminho crítico (`index.html:134`): `loading="lazy"`/poster estático e, com aprovação, troca por MP4/WebM ≤ 2 MB — transferência inicial ≤ 2,0 MB medida no Lighthouse mobile local.
- [ ] **B6 (P1, O2)** Configurar cobertura: `@vitest/coverage-v8` 5.0.5 com `thresholds` e `coverage.py` no job `python` — CI falha abaixo das metas de O2; relatório publicado como artefato.
- [ ] **B7 (P1, O6)** Acentuação e PT-BR em `index.html`, `src/shared/errors.ts`, `src/ui/monitorView.ts` e README — `grep -c '[áéíóúãõç]' index.html` > 0 e nenhum termo sem acento na lista revisada; testes de texto atualizados.
- [ ] **B8 (P2, O6)** Shell visual do padrão (IBM Plex Sans, tokens, rodapé A.9) no lugar de Inter e da paleta própria (`src/styles.css:25-38`) — axe 0 serious/critical em 320/768/1280; screenshots 375/1440 anexados ao PR.
- [ ] **B9 (P2, O5)** `ruff format` nos 4 arquivos e passo `ruff format --check .` no `ci.yml` — comando retorna 0 local e no CI.
- [ ] **B10 (P2, O3)** Self-host do WASM e dos dois modelos em `public/` (se aprovado) e remoção de jsDelivr/Google Storage da CSP — e2e da câmera sintética verde com a CSP nova; `git grep cdn.jsdelivr src vercel.json` vazio.
- [ ] **B11 (P2, O1)** Alinhar Node local e `engines` (`.nvmrc` com 24) e documentar no README — `npm ci` sem `EBADENGINE` em Node 24.
- [ ] **B12 (P2, O2)** Teste unitário do `drawDemo` garantindo rótulo "ilustrativo" nas caixas da demo (`src/vision/overlayRenderer.ts:103-130`) — Vitest com canvas falso verifica o texto desenhado.
- [ ] **B13 (P3, O1)** README com link da demo e tabela "o que a web mede / o que só o Python mede" — links verificados e tabela coerente com `src/vision` e `dms/`.

## 7. Radar técnico

| Data | Item | Fonte | Veredito (adotar / avaliar / descartar) | Motivo |
| --- | --- | --- | --- | --- |
| 2026-10-23 | `@vitest/coverage-v8` 5.0.5 (mesma versão do Vitest instalado) | https://vitest.dev/guide/coverage | adotar | Habilita O2 sem trocar o runner; versão alinhada evita conflito de peer. |
| 2026-10-23 | TypeScript 7.0.2 (compilador nativo) disponível no npm | https://devblogs.microsoft.com/typescript/ | avaliar | Major: typecheck mais rápido, mas exige validar `typescript-eslint` 8.x; só com aprovação. |
| 2026-10-23 | Vite 8.3.5 (patch sobre 8.3.4) | https://github.com/vitejs/vite/releases | adotar | Patch dentro da mesma major; baixo risco, cobrir com `npm run validate`. |
| 2026-10-23 | Ultralytics 8.4.184 (pin atual 8.4.178) | https://pypi.org/project/ultralytics/ | avaliar | Patches frequentes; atualizar só junto com o CI e o teste headless com vídeo. |
| 2026-10-23 | Python 3.12 em fase só de correções de segurança | https://devguide.python.org/versions/ | avaliar | Planejar 3.13 quando torch, mediapipe e opencv fixados tiverem wheels validados; hoje `requires-python <3.13`. |
| 2026-10-23 | MediaPipe Tasks Vision em Web Worker / `OffscreenCanvas` | https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js | avaliar | Já previsto no `MASTER_EVOLUTION_PLAN.md`; libera a main thread durante a inferência. |

## 8. Ideias de features

| # | Feature | Valor para o usuário/portfólio | Risco | Status (proposta / aprovada / implementada / descartada) |
| --- | --- | --- | --- | --- |
| F1 | Olhar por íris (landmarks 468-477) para "eyes off road" | Sinal além da pose da cabeça; demonstra geometria aplicada com testes de série | Médio (ruído e calibração por pessoa) | proposta |
| F2 | Calibração por motorista nos primeiros 10-30 s (EAR/MAR base) | Reduz falsos positivos e mostra definição de baseline estatístico | Médio (precisa espelhar no Python) | proposta |
| F3 | Vídeo de amostra rodando o pipeline real no lugar do GIF | Demonstração sem câmera com medições reais, não roteirizadas | Médio (depende de vídeo com direito de uso) | proposta |
| F4 | Linha do tempo de atenção de 60 s com faixa de limiar e lista textual equivalente | Elemento visual central, acessível e verificável | Baixo | proposta |
| F5 | Inferência em Web Worker | HUD fluido em CPU modesta; demonstra arquitetura de front-end de alto desempenho | Médio (CSP `worker-src`, transferência de frames) | proposta |

## 9. Decisões pendentes do dono

- Remover o modo "Review" ou mantê-lo renomeado como cenário sintético de risco?
- Self-host do WASM e dos modelos do MediaPipe (fecha a CSP, aumenta o deploy) ou manter jsDelivr e Google Storage?
- Converter `public/assets/demo.gif` (6,0 MB) para vídeo ≤ 2 MB? Existe vídeo próprio com direito de uso para F3?
- Speed Insights continua sem aviso de consentimento, citado apenas na nota de privacidade?
- Adotar Conventional Commits a partir deste ciclo e PRs menores (os dois maiores commits somam mais de 22 mil linhas)?
- Upgrade para TypeScript 7 (major) entra neste ciclo?

## 10. Registro semanal

| Data | Auditoria | Mudanças (commits) | Métricas | Próximo foco |
| --- | --- | --- | --- | --- |
| 2026-10-23 | Documento inicial e baseline | — | ver seção 2 | O1/B1-B2 |
