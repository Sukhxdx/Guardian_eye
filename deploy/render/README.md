# Deploy GuardianEye on Render

GuardianEye is a FastAPI + Gradio service. Render injects a dynamic `PORT`; the
entrypoint maps that to `GUARDIANEYE_PORT`.

## Plan size

Use **Standard (2 GB RAM)** or larger. Free and Starter (512 MB) will typically
run out of memory when importing PyTorch, Ultralytics, and Gradio.

## One-time setup

1. Push this repository to **GitHub or GitLab** (Render does not pull from Cursor Origin).
2. Open [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint**.
3. Select the repo. Render reads `render.yaml` and builds `Dockerfile.render`.
4. Set optional env vars `HF_MODEL_REPO` / `HF_TOKEN` if you host weights on the Hub.
5. After deploy, open `https://<service>.onrender.com/health` and `/dashboard`.

### Manual web service (without Blueprint)

- **Runtime:** Docker  
- **Dockerfile path:** `Dockerfile.render`  
- **Health check path:** `/health`  
- **Instance type:** Standard or higher  

## Local Docker check (same image Render builds)

```bash
docker build -f Dockerfile.render -t guardianeye-render .
docker run --rm -p 8742:8742 -e PORT=8742 guardianeye-render
curl -s http://127.0.0.1:8742/health
```

## Endpoints

| Path | Purpose |
| --- | --- |
| `/health` | Liveness |
| `/model-info` | Which checkpoints loaded |
| `/dashboard` | Gradio ranger UI |
| `/events` | Audit log API |

Proof-of-concept. Indicators, not confirmed poaching. Without weights under
`models/` or a Hub repo, `/model-info` reports unreliable evidence by design.
