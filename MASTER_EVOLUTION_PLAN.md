# Master Evolution Plan

Documento executivo da evolucao do Driver Monitoring System: de onde o projeto saiu, onde esta, para onde deve ir e como implementar os proximos passos. Os blocos de codigo da secao 4 sao trechos reais do repositorio; os da secao 3 sao guias de implementacao propostos (ainda nao implementados).

## 1. Sumario executivo

| Dimensao | Estado legado | Estado atual |
|---|---|---|
| Web | `app.js` em JavaScript sem tipos, build com script proprio | TypeScript estrito + Vite, camadas `domain/app/vision/infra/ui`, composition root com injecao de dependencias |
| Regras de score | Duplicadas e divergentes entre web e Python | `shared/scoring-config.json` como fonte unica; 14 cenarios de paridade verificados nos dois runtimes |
| Deteccao de fadiga | EAR com sonolencia/microssono | + PERCLOS (60 s), bocejo por MAR e distracao acumulada Euro NCAP (VATS) |
| Python | Dependencias sem versao fixa, backend MediaPipe legado | Python 3.12 com versoes fixadas, apenas Tasks API, pipeline com Protocols, erros com exit codes |
| Performance (engine, CPU 720p) | YOLO em todo frame: 16.8 FPS | YOLO/maos a cada 3 frames: 40.6 FPS (2.4x) |
| Bundle web | Sem medicao registrada | Bundle inicial de 14 kB gzip; MediaPipe (46 kB gzip + WASM) carregado sob demanda ao ativar a camera |
| Testes | unittest basico | 48 unittest + 50 Vitest + 9 Playwright E2E (inclui axe-core WCAG 2.2 A/AA); CI com lint, typecheck, paridade, build e E2E |
| Seguranca e privacidade | Headers so na Vercel | Mesmos headers no `vite preview`; telemetria do MediaPipe bloqueada pela CSP; CSV protegido contra injecao de formula |
| Produto | Dashboard de demonstracao | Dashboard com PERCLOS, bocejos e VATS; celular detectado tambem no navegador; exportacao JSON/CSV local |

## 2. Melhorias arquiteturais de longo prazo

1. **Inferencia fora da main thread.** Mover `FaceLandmarkerService` para um Web Worker com `OffscreenCanvas` e `transferControlToOffscreen`. O pipeline de dominio ja e puro (sem DOM) e pode rodar no worker; a UI recebe apenas `PipelineResult` serializado. Ganho: HUD a 60 fps mesmo com inferencia em CPU.
2. **Contrato de sinais versionado.** `FrameSignals` e o schema de relatorio (`schemaVersion: 1`) sao a fronteira entre visao e regras. Publicar como JSON Schema em `shared/` permite que o engine Python e um futuro backend validem o mesmo formato.
3. **Plugins de detector.** No Python os detectores ja sao `Protocol`s; no web, introduzir uma interface `SignalSource` (camera, demo, video gravado, replay de relatorio) e registrar fontes no composition root.
4. **Regras como dados.** A tabela de penalidades ja esta em JSON; o proximo passo e descrever cada evento como `{ sinal, comparacao, limiar, duracao, penalidade }` e gerar os dois scorers a partir dessa tabela, reduzindo codigo espelhado.
5. **Calibracao por pessoa.** EAR e MAR variam por anatomia. Um perfil de baseline (primeiros 30 s) deve escalar os limiares, sem alterar a fonte compartilhada.
6. **Observabilidade sem dados pessoais.** Manter analytics opt-in, apenas eventos agregados (`attention_event`, `report_exported`), nunca landmarks nem imagem.

## 3. Guias taticos para as novas features

### 3.1 Celular no navegador (implementado)

Implementado em `src/vision/phoneDetector.ts`, `src/vision/visionRuntime.ts` e `src/domain/spatial.ts`:

- **Um runtime WASM para todas as tasks:** `loadVisionRuntime()` carrega o modulo e o fileset uma vez; Face Landmarker e Object Detector compartilham.
- **Modelo:** EfficientDet-Lite0 float16, `categoryAllowlist: ["cell phone"]`, limiar 0.3. Comparacao em 6 fotos publicas (delegate CPU, limiar 0.2): Lite0 int8 achou 2/6 (114 ms), Lite0 float16 3/6 (121 ms), Lite2 2/6 com mais confianca (209-398 ms). Em 10 outras fotos o Lite0 float16 achou o celular em 7.
- **Custo controlado:** inferencia no maximo a cada 400 ms, resultado reutilizado e descartado apos 1 s; carregamento em segundo plano e opcional (se falhar, o modo Camera segue sem sinal de celular).
- **Mesma regra do Python:** caixas mapeadas para o espaco do canvas (com `object-fit: cover` e espelhamento) e `isPhoneNearFace` validado contra `shared/spatial-cases.json` (127 casos gerados pelo Python).
- **E2E:** uma foto real (CC BY 2.0) e servida como camera via `canvas.captureStream`; o teste exige o evento "Celular" no pipeline completo.

Proximos passos: mover a inferencia para um Web Worker e reavaliar o Lite2 com delegate GPU.

### 3.2 Olhar (gaze) por iris

O Face Landmarker ja entrega 478 pontos; 468-477 sao as iris. Razao horizontal da iris entre os cantos do olho:

```ts
// src/domain/gaze.ts (proposta)
import { distance, type Point } from "./geometry";

/** 0 = canto externo, 1 = canto interno; ~0.5 olhando para frente. */
export function irisRatio(landmarks: readonly Point[], outer: number, inner: number, iris: number): number | null {
  const a = landmarks[outer];
  const b = landmarks[inner];
  const c = landmarks[iris];
  if (!a || !b || !c) return null;
  const width = distance(a, b);
  return width === 0 ? null : distance(a, c) / width;
}
```

Combinar com a pose: "eyes off road" quando pose ou iris passam do limite. Isso cobre o caso "Lizard" do Euro NCAP (olhos se movem, cabeca parada). Adicionar o sinal ao `RollingDutyCycle` do VATS.

### 3.3 Calibracao por motorista

```ts
// proposta: nos primeiros 30 s com rosto presente
const baselineEar = median(openEarSamples);
const calibrated = { ...config, earThreshold: Math.min(config.earThreshold, baselineEar * 0.75) };
pipeline.reconfigure(calibrated);
```

`MonitoringPipeline.reconfigure` ja preserva timers e score, entao a calibracao entra sem reiniciar a sessao.

### 3.4 Alerta sonoro opcional (implementado)

`src/domain/alertPolicy.ts` decide quando tocar: so no inicio de um evento, com cooldown de 2.5 s, e um alerta urgente pode interromper o cooldown de um aviso. `src/infra/audioAlert.ts` toca com Web Audio, desligado por padrao e ligado pelo toggle "Som" (gesto do usuario, exigido pela politica de autoplay). Sonolencia e microssono tocam 3 tons (1050/1400/1050 Hz); distracao e celular, um tom de 740 Hz. Fadiga, bocejo e celular suspeito ficam apenas visuais. Um E2E verifica o padrao urgente e a faixa de 200-8000 Hz.

### 3.5 Relatorio no engine Python (implementado)

`python main.py --source video.mp4 --report sessao.json` grava o mesmo schema da exportacao web (`schemaVersion: 1`, com `source.sensitivity: null` porque o engine nao tem o controle de sensibilidade). `dms/session.py` espelha `SessionStats` e o resumo de sessao faz parte dos cenarios de paridade.

### 3.6 Acessibilidade continua (implementado)

`tests/e2e/a11y.spec.ts` roda axe-core (WCAG 2.2 A/AA) no build de producao e testa operacao por teclado. Proximo passo: auditar tambem o estado do modo Camera e o contraste do overlay sobre o video.

## 4. Blocos criticos refatorados, por dominio

### 4.1 Dominio: janela deslizante identica nos dois runtimes

`src/domain/rolling.ts` (espelhado por `dms/rolling.py`):

```ts
add(timestampS: number, active: boolean): void {
  let span = 0;
  let activeSpan = 0;
  const previous = this.samples[this.samples.length - 1];
  if (previous) {
    // A long gap (face lost, dropped frames) only counts up to maxGapS.
    span = Math.min(Math.max(timestampS - previous.t, 0), this.maxGapS);
    activeSpan = previous.active ? span : 0;
  }
  this.samples.push({ t: timestampS, active, span, activeSpan });
  this.total += span;
  this.activeTotal += activeSpan;
  // ...descarta amostras fora da janela subtraindo a contribuicao da nova primeira amostra
}
```

### 4.2 Dominio: regras novas no scorer

`src/domain/attention.ts`:

```ts
const { perclos } = input.eyeState;
if (perclos !== null && perclos >= config.perclosThreshold) {
  penalty += config.penaltyFatigue;
  events.push("Fadiga");
}
if (input.mouthState?.yawning) {
  penalty += config.penaltyYawn;
  events.push("Bocejo");
}
// ...
if (pose !== null) {
  this.offroadWindow.add(input.timestampS, offroad);
}
if (this.offroadWindow.activeS >= config.vatsOffroadLimitS) {
  penalty += config.penaltyVats;
  events.push("Distracao acumulada");
}
```

### 4.3 Configuracao: fonte unica

`dms/config.py`:

```python
def load_shared_scoring(path: Path = SHARED_SCORING_CONFIG) -> Dict[str, float]:
    """Scoring rules shared with the web app; the JSON is the single source of truth."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Nao foi possivel ler {path}: {exc}") from exc
    ...

@dataclass
class DMSConfig:
    ear_threshold: float = _SCORING["ear_threshold"]
    perclos_threshold: float = _SCORING["perclos_threshold"]
```

### 4.4 Performance: detectores em passo

`dms/pipeline.py`:

```python
def _run_detectors(self, frame, timestamp_ms):
    """Phone/hand detectors run every ``detector_interval`` frames; in between the last result is reused."""
    if self._frame_index % self.detector_interval == 0:
        if self.phone_detector is not None:
            self._last_detections = self.phone_detector.detect(frame)
        if self.hand_source is not None:
            self._last_hand_boxes = self.hand_source.detect(frame, timestamp_ms)
    self._frame_index += 1
    return self._last_detections, self._last_hand_boxes
```

A proximidade celular-rosto continua sendo recalculada a cada frame com a bbox atual do rosto.

### 4.5 Infra: headers de producao no preview

`vite.config.ts`:

```ts
function productionHeaders(): Record<string, string> {
  const config = JSON.parse(readFileSync(resolve(import.meta.dirname, "vercel.json"), "utf8"));
  const rule = config.headers?.find((entry) => entry.source === "/(.*)");
  // HSTS on localhost would pin the browser to HTTPS for every localhost port.
  const headers = (rule?.headers ?? []).filter((header) => header.key !== "Strict-Transport-Security");
  return Object.fromEntries(headers.map((header) => [header.key, header.value]));
}
```

### 4.6 Seguranca: CSV sem injecao de formula

`src/domain/report.ts`:

```ts
function csvCell(value: string | number): string {
  const text = String(value);
  const safe = /^[=+\-@]/.test(text) ? `'${text}` : text;
  return /[",\n]/.test(safe) || safe !== text ? `"${safe.replaceAll('"', '""')}"` : safe;
}
```

### 4.7 Dependencias: versao JS e WASM sincronizadas

`tests/web/faceLandmarker.test.ts` garante que `TASKS_VISION_VERSION` (URL do WASM no CDN) e igual a versao do pacote no `package.json`; uma divergencia so apareceria em tempo de execucao.

## 5. Riscos e limites conhecidos

- **Limiares heuristicos:** PERCLOS 15% em 60 s e MAR 0.6 por 1.5 s nao tem referencia normativa unica; devem ser calibrados com dados (ex.: YawDD).
- **Euro NCAP x projeto:** o protocolo pede deteccao de distracao longa em 3-4 s; o projeto usa 1.5 s (mais sensivel, adequado para demo). O VATS segue o protocolo (10 s em 30 s).
- **Celular no web:** o EfficientDet-Lite0 tem recall limitado (celular encontrado em cerca de metade a dois tercos das fotos testadas); amostra pequena, nao e uma avaliacao formal.
- **TypeScript 7:** bloqueado ate o `typescript-eslint` suportar.
- **Telemetria do MediaPipe 1.1.x:** bloqueada pela CSP em producao; em ambientes sem CSP (ex.: `vite dev`) o envio de contadores pode ocorrer.
- **Nao e um sistema automotivo certificado:** projeto educacional e de portfolio.
