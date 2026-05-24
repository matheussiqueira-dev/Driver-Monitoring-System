# Driver Monitoring System (DMS)

Sistema de Monitoramento do Motorista em tempo real usando Visao Computacional e IA. Detecta sonolencia, distracao, direcao do olhar e uso de celular ao volante com Face Mesh, Eye Aspect Ratio, Head Pose Estimation e Object Detection, gerando um Attention Score continuo.

![Demo](assets/demo.gif)

## Desenvolvedor

- Desenvolvido por Matheus Siqueira
- Portfolio oficial: [www.matheussiqueira.dev](https://www.matheussiqueira.dev)

## Entregas do projeto

- Pipeline Python/OpenCV para processamento em tempo real.
- Overlay visual com malha facial, boxes, metricas, score e alertas.
- Interface web estatica para portfolio e deploy na Vercel.
- Dashboard inteligente com KPIs, tendencias, eventos e insights.
- Regras mais robustas para reduzir falsos positivos de celular.
- Configuracao `vercel.json` com cache de assets e rotas limpas.

## Principais recursos

- Face Mesh (MediaPipe) com 468 landmarks e suavizacao por EMA.
- Eye Aspect Ratio (EAR) para piscadas, sonolencia e microssono.
- Head Pose Estimation com yaw, pitch e roll via `solvePnP`.
- Deteccao de celular por YOLO (Ultralytics).
- Fallback de maos com MediaPipe Hands.
- Score de atencao 0-100 com suavizacao temporal.
- Dashboard responsivo preparado para integracao futura com dados reais.

## Arquitetura

```text
Driver-Monitoring-System/
  assets/
    demo.gif
  dms/
    attention.py
    camera.py
    config.py
    detection.py
    ear.py
    face_mesh.py
    head_pose.py
    mediapipe_utils.py
    spatial.py
    utils.py
    visualization.py
  tests/
  app.js
  index.html
  main.py
  package.json
  requirements.txt
  styles.css
  vercel.json
```

## Pipeline Python

1. Captura de frame por camera ou video.
2. Face Mesh para landmarks faciais.
3. Calculo de EAR e estimativa de pose da cabeca.
4. YOLO para celular e MediaPipe Hands como fallback.
5. Fusao de sinais no `AttentionScorer`.
6. Renderizacao do overlay em OpenCV.
7. Score continuo, barra animada e alertas.

## Como rodar o DMS em tempo real

Recomendado: Python 3.9 a 3.12. Algumas versoes do MediaPipe podem nao oferecer wheels para Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py --source 0
```

Selecionar camera pelo nome no Windows:

```bash
python -m pip install pygrabber
python main.py --list-cameras
python main.py --camera-name Brio
```

Rodar com video:

```bash
python main.py --source path/para/video.mp4
```

Opcoes uteis:

```bash
python main.py --source 0 --no-mirror
python main.py --weights pesos_custom.pt --device cuda
python main.py --no-yolo --no-hands --no-mesh
```

## Interface web e Vercel

A interface web e estatica e pode ser publicada diretamente na Vercel. Ela usa o `assets/demo.gif` e um overlay em canvas para demonstrar o comportamento visual do sistema, com dados simulados desacoplados para KPIs e insights.

Execucao local:

```bash
python -m http.server 3000
```

Abra:

```text
http://localhost:3000
```

Validacao de build estatico:

```bash
npm run build
```

O build gera a pasta `dist/`, que e a saida configurada para a Vercel. O arquivo `.vercelignore` envia apenas a superficie web estatica para deploy e impede que `main.py` seja interpretado como Python Function.

Deploy na Vercel:

```bash
vercel
vercel --prod
```

Variaveis de ambiente:

```text
Nenhuma variavel e obrigatoria para a interface web estatica.
```

## Dashboard

O dashboard inclui:

- Score medio e tendencia dos ultimos segundos.
- Eventos criticos e classificacao de risco.
- EAR medio e FPS medio.
- Lista de eventos ativos.
- Insight automatico contextual.
- Estrutura pronta para substituir mock data por API, websocket ou telemetria real.

## Tratamento de desafios

- Iluminacao variavel: landmarks robustos e ajuste de `detection_confidence`/`tracking_confidence`.
- Oculos escuros: reduzir dependencia do EAR e priorizar head pose.
- Oclusao parcial: suavizacao com EMA nos landmarks.
- Falsos positivos de celular: filtro por area minima, proximidade do rosto e pitch para baixo.
- Sensibilidade vs. robustez: ajuste de `drowsy_time_s`, `offroad_time_s` e penalidades em `DMSConfig`.

## Testes

```bash
python -m unittest discover -s tests
```

Os testes atuais cobrem regras de score e filtro espacial de celular. Para evolucao, recomenda-se adicionar testes E2E da interface e cenarios com videos curtos anotados.

## Aviso

Este projeto e para fins educacionais, demonstrativos e de portfolio. Sistemas automotivos reais exigem testes extensivos, redundancia e certificacoes especificas.
