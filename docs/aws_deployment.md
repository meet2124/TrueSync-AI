# TrueSync AI — AWS Deployment Guide

**Target region:** `ap-south-1` (Mumbai)  
**Architecture:** EC2 → FastAPI/Uvicorn → WebSocket → Streamlit

> **Do NOT deploy automatically.** This guide documents the steps required for manual deployment.

---

## Architecture

```
User (Browser)
    │  HTTPS / WSS
    ▼
AWS EC2 (ap-south-1)
    ├── FastAPI backend  (port 8000, Uvicorn)
    │       └── /ws/session  (WebSocket)
    │       └── /health      (REST probe)
    └── Streamlit frontend (port 8501)
```

---

## 1. EC2 Instance Recommendation

| Setting | Value |
|---------|-------|
| **AMI** | Ubuntu 22.04 LTS |
| **Instance type** | `t3.medium` (2 vCPU, 4 GB RAM) minimum; `t3.large` for live demo |
| **Region** | `ap-south-1` |
| **Storage** | 20 GB gp3 |
| **Key pair** | Create before launch, store `.pem` securely |

---

## 2. Security Group Rules

| Type | Protocol | Port | Source | Purpose |
|------|----------|------|--------|---------|
| SSH | TCP | 22 | Your IP only | Admin access |
| Custom TCP | TCP | 8000 | `0.0.0.0/0` | FastAPI / WebSocket |
| Custom TCP | TCP | 8501 | `0.0.0.0/0` | Streamlit dashboard |
| HTTP | TCP | 80 | `0.0.0.0/0` | Optional: nginx reverse proxy |
| HTTPS | TCP | 443 | `0.0.0.0/0` | Optional: SSL termination |

---

## 3. Server Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ and dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip git

# Install OpenCV system dependencies (required for MediaPipe)
sudo apt install -y libgl1-mesa-glx libglib2.0-0

# Clone repository
git clone https://github.com/meet2124/TrueSync-AI.git
cd TrueSync-AI

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env   # Edit: BACKEND_HOST=0.0.0.0, WEBSOCKET_URL=ws://<EC2_IP>:8000/ws/session
```

---

## 4. Running the Services

### Option A — Direct (development/demo)

```bash
# Terminal 1: Backend
cd /home/ubuntu/TrueSync-AI
source .venv/bin/activate
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --workers 1

# Terminal 2: Frontend
cd /home/ubuntu/TrueSync-AI
source .venv/bin/activate
streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

### Option B — systemd services (production)

Create `/etc/systemd/system/truesync-backend.service`:
```ini
[Unit]
Description=TrueSync AI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/TrueSync-AI
ExecStart=/home/ubuntu/TrueSync-AI/.venv/bin/uvicorn backend.server:app \
    --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
Environment=LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/truesync-frontend.service`:
```ini
[Unit]
Description=TrueSync AI Frontend
After=truesync-backend.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/TrueSync-AI
ExecStart=/home/ubuntu/TrueSync-AI/.venv/bin/streamlit run frontend/app.py \
    --server.address 0.0.0.0 --server.port 8501 --server.headless true
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable truesync-backend truesync-frontend
sudo systemctl start truesync-backend truesync-frontend
sudo systemctl status truesync-backend truesync-frontend
```

---

## 5. Environment Variables for EC2

Edit `.env` on the server:

```env
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
WEBSOCKET_URL=ws://<YOUR_EC2_PUBLIC_IP>:8000/ws/session
ENGINE_MODE=mock
CORS_ORIGINS=*
MAX_PAYLOAD_BYTES=5242880
WORKER_THREADS=4
LOG_LEVEL=INFO
AWS_REGION=ap-south-1
```

> Replace `<YOUR_EC2_PUBLIC_IP>` with the EC2 instance's public IPv4 address.

---

## 6. Health Check Verification

```bash
# From local machine
curl http://<EC2_IP>:8000/health

# Expected response:
# {"status":"ok","uptime_seconds":...,"version":"production-v1","engine":"production",...}
```

Configure AWS **EC2 Target Group** health check to hit `GET /health` on port 8000.

---

## 7. Optional: CloudWatch Logging

If you want logs in AWS CloudWatch:

1. Install the CloudWatch agent on the EC2 instance
2. Set `LOG_LEVEL=INFO` in `.env`
3. Configure the agent to tail the Uvicorn stdout journal

> Do **not** log raw audio/video data or biometric scores at DEBUG level in production.  
> Set `LOG_LEVEL=WARNING` on production to reduce noise.

---

## 8. WebSocket over HTTPS (WSS)

For `wss://` (required if frontend is served over HTTPS):

1. Use **nginx** as a reverse proxy with SSL termination  
2. Proxy `/ws/` → `ws://localhost:8000/ws/`  
3. Use AWS **Certificate Manager** for the TLS certificate  
4. Update `.env`: `WEBSOCKET_URL=wss://your-domain.com/ws/session`

A minimal nginx config snippet:
```nginx
location /ws/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "Upgrade";
    proxy_read_timeout 86400;
}
```

---

## 9. What Is Ready

- [x] Backend binds to `0.0.0.0` (not localhost) via `BACKEND_HOST` env var
- [x] `WEBSOCKET_URL` is configurable — no hardcoded localhost
- [x] `CORS_ORIGINS` configurable for production restriction
- [x] `MAX_PAYLOAD_BYTES` configurable
- [x] `LOG_LEVEL` configurable
- [x] `AWS_REGION` documented in `.env.example`
- [x] `/health` endpoint is safe for ALB health checks
- [x] systemd service files documented
- [x] No AWS credentials in code or `.env.example`
- [x] No biometric data uploaded to S3 (not required by current architecture)

## 10. Known Limitations / Remaining Steps

- [ ] SSL/TLS not configured — requires nginx + ACM for WSS
- [ ] No auto-scaling — single EC2; add ALB + ASG if needed
- [ ] Camera/mic access requires HTTPS in browsers (localhost exception exists)
- [ ] `ENGINE_MODE=production` requires real ML models (Phase 2)
- [ ] No authentication on WebSocket endpoint (add API key header if needed)
