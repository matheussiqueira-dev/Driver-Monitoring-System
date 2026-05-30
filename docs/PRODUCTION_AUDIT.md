# Production Audit

Projeto: Driver Monitoring System  
Responsavel: Matheus Siqueira  
Portfolio: https://www.matheussiqueira.dev  
LinkedIn: https://www.linkedin.com/in/matheussiqueira-dev

## Diagnostico Executivo

O projeto esta estruturado como uma aplicacao web estatica profissional para portfolio e deploy na Vercel, acompanhada por uma engine Python/OpenCV local para processamento real de Computer Vision. Essa separacao e intencional: a superficie web entrega a experiencia de produto, enquanto o pipeline Python preserva a demonstracao tecnica de camera, MediaPipe, EAR, Head Pose, YOLO e Attention Score sem forcar execucao serverless inadequada na Vercel.

## Classificacao de Problemas

### Critico

- **Vercel detectando preset Python no projeto remoto.**  
  Status: mitigado no repositorio.  
  Evidencia: `vercel.json` define `framework: null`, `buildCommand: node scripts/build-static.js`, `outputDirectory: dist`, e `.vercelignore` impede que `main.py` seja enviado como Function.

### Alto

- **Risco de regressao no tracking facial web.**  
  Status: mitigado.  
  Evidencia: `app.js` usa MediaPipe Tasks Vision em `runningMode: "VIDEO"`, loop continuo, mapeamento canvas/video com `object-fit: cover`, espelhamento e debug de `frameId`, landmarks e FPS.

- **Observabilidade dependente de configuracao externa.**  
  Status: mitigado.  
  Evidencia: `analytics.js` cria fila de eventos, captura Web Vitals e erros, e carrega scripts Vercel somente via meta tag configurada.

### Medio

- **Mock data no dashboard.**  
  Status: aceito e documentado.  
  Justificativa: nao ha backend de sessoes ainda. A arquitetura esta desacoplada para evoluir para WebSocket/API sem reescrever UI.

- **Speed Insights limitado por plano Vercel.**  
  Status: parcialmente mitigado.  
  Evidencia: o script de Speed Insights e carregado quando disponivel; a coleta efetiva depende da configuracao/limites da conta Vercel.

### Baixo

- **Projeto nao usa React/Next.js.**  
  Status: decisao arquitetural mantida.  
  Justificativa: converter para Next.js aumentaria complexidade sem necessidade para uma superficie estatica de portfolio. A aplicacao atual atende Vercel, PWA, SEO e analytics com menor risco operacional.

## Arquitetura Atual

```text
Browser
  index.html
  styles.css
  app.js
  analytics.js
  pwa.js
  service-worker.js
  manifest.webmanifest

Python local
  main.py
  dms/
    face_mesh.py
    overlay.py
    head_pose.py
    ear.py
    detection.py
    attention.py

Deploy
  scripts/build-static.js -> dist/
  vercel.json
  .vercelignore
```

## Validacoes Obrigatorias

```bash
npm run validate
python -m unittest discover -s tests
```

Checklist validado:

- Build estatico gera `dist/`.
- JS modules passam em `node --check`.
- Testes Python cobrem score, filtro espacial e overlay tracking.
- PWA possui manifest, icones e service worker.
- SEO possui canonical, Open Graph, Twitter Cards, robots, sitemap e structured data.
- Creditos de Matheus Siqueira aparecem no footer e na secao Sobre.
- Vercel usa output estatico e nao precisa executar Python.

## Seguranca

Controles em `vercel.json`:

- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy` restringindo camera ao proprio app
- `X-Frame-Options: DENY`
- `Strict-Transport-Security`
- `Content-Security-Policy`

## Observabilidade

Eventos instrumentados:

- `source_mode_changed`
- `overlay_layer_toggled`
- `camera_ready`
- `camera_start_failed`
- `face_mesh_ready`
- `face_mesh_error`
- `web_vital`
- `client_error`
- `pwa_service_worker_ready`
- `pwa_service_worker_error`

## Recomendacoes Futuras

- Ajustar o preset remoto da Vercel para `Other`/static nas Project Settings.
- Adicionar testes E2E com Playwright versionados no repositorio.
- Criar API opcional para persistir sessoes e substituir mock data do dashboard.
- Integrar modelo customizado YOLO com classes `cell phone` e `hand`.
- Adicionar pipeline de release com GitHub Actions para `npm run validate` e testes Python.
