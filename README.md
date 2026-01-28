# Driver Monitoring System (DMS)

Sistema de Analise de Atencao do Motorista em tempo real usando camera frontal. Detecta sonolencia, distracao, movimentos de cabeca e uso de celular, gerando um Attention Score continuo e visual.

## Principais recursos
- Face Mesh (MediaPipe) com 468 landmarks e suavizacao.
- Eye Aspect Ratio (EAR) para piscadas e sonolencia.
- Head Pose Estimation (yaw, pitch, roll) via solvePnP.
- Deteccao de celular por YOLO (Ultralytics).
- Score de atencao (0-100) com suavizacao temporal.
- Overlay visual com malha facial, boxes, metricas e barra animada.

## Pipeline
1. Captura de frame (camera ou video)
2. Face Mesh
3. EAR + Head Pose
4. YOLO (celular) + Hands (fallback)
5. Fusao de sinais
6. Attention Score
7. Renderizacao

## Requisitos
- Python 3.9+
- Webcam ou video de teste

## Instalacao
```bash
pip install -r requirements.txt
```

> Observacao: `ultralytics` baixa automaticamente os pesos do YOLO na primeira execucao.
> Em versoes recentes do MediaPipe, os modelos `.task` de Face/Hand Landmarker sao baixados automaticamente para `models/` na primeira execucao.

## Como rodar
Camera padrao:
```bash
python main.py --source 0
```

Video:
```bash
python main.py --source path/para/video.mp4
```

Sem espelhamento (camera traseira ou video):
```bash
python main.py --source 0 --no-mirror
```

YOLO customizado (incluindo classe `hand`):
```bash
python main.py --weights pesos_custom.pt --device cuda
```

Desativar modulos pesados (para ganho de FPS):
```bash
python main.py --no-yolo --no-hands --no-mesh
```

## Controles
- `q` ou `ESC` para sair.

## Parametros importantes
- `DMSConfig` em `dms/config.py` define thresholds e pesos do score.
- EAR abaixo de `ear_threshold` por `drowsy_time_s` aciona sonolencia.
- Desvio prolongado de yaw/pitch acima de `yaw_threshold`/`pitch_threshold` gera penalidade.
- Celular detectado adiciona penalidade alta ao score.

## Visualizacao
A interface desenha:
- Malha facial (Face Mesh)
- Boxes de deteccao
- EAR, yaw/pitch/roll, FPS
- Score e barra animada
- Alertas em tempo real

## Tratamento de desafios
- **Iluminacao variavel**: usa landmarks robustos; ajuste `min_detection_confidence` e `min_tracking_confidence`.
- **Oculos escuros**: reduzir dependencia do EAR e priorizar head pose; aumentar tolerancia no limiar de olhos.
- **Oclusao parcial**: suavizacao com EMA nos landmarks; retoma rapidamente quando a face volta.
- **Falsos positivos de celular**: filtra por area minima e proximidade do rosto; combine com pitch para baixo.
- **Sensibilidade vs robustez**: ajuste `drowsy_time_s`, `offroad_time_s` e penalidades em `DMSConfig`.

## Extensoes sugeridas
- Modelo customizado com classe `hand` no YOLO.
- Classificador de distracao visual com olhar (gaze estimation).
- Detecao de bocejo por abertura de boca.
- Adaptacao automatica de thresholds por usuario.

## Estrutura do projeto
```
Driver Monitoring System/
  dms/
    attention.py
    config.py
    detection.py
    ear.py
    face_mesh.py
    head_pose.py
    utils.py
    visualization.py
  main.py
  requirements.txt
  README.md
```

## Notas de desempenho
- Em maquinas sem GPU, use `yolov8n.pt` para manter FPS acima de 20.
- Desative o YOLO com `--no-yolo` para focar apenas em sonolencia/pose.

---

**Aviso**: Este projeto e para fins educacionais e de demonstracao. Em sistemas automotivos reais, sao necessarios testes extensivos, redundancia e certificacoes.
