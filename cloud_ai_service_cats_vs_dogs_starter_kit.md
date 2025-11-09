# Cloud-Hosted AI Service (Cats vs Dogs) — Starter Kit

An end-to-end project that trains a lightweight image classifier, serves it via **FastAPI**, packages with **Docker**, deploys to the **cloud free tier**, and auto-updates via **GitHub Actions CI/CD**.

---

## 📁 Repository Structure
```
cloud-ai-cats-dogs/
├─ README.md
├─ requirements.txt
├─ .dockerignore
├─ Dockerfile
├─ artifacts/
│  ├─ model.pt                # created after training
│  └─ class_indices.json      # created after training
├─ app/
│  └─ main.py                 # FastAPI app
├─ src/
│  ├─ train.py                # training script
│  └─ utils.py                # small helpers
├─ tests/
│  └─ test_api.py             # simple API tests
└─ .github/workflows/
   └─ cicd.yml                # build + push image + (optional) deploy via SSH
```

---

## ✅ Prerequisites
- Python 3.11
- pip + virtualenv (recommended)
- Docker Desktop (free)
- Git + GitHub account
- (For CI/CD + deploy) A free-tier VM (e.g., **AWS EC2 t2.micro**) and SSH key

---

## 1) Quickstart — Local Dev

### 1.1 Create & activate venv
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 1.2 Install deps (without torch)
```bash
pip install -r requirements.txt
# Install CPU-only PyTorch (Linux/macOS/Windows):
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

### 1.3 Prepare data
Use any cats/dogs dataset arranged as:
```
DATA_DIR/
  train/
    cat/*.jpg
    dog/*.jpg
  val/
    cat/*.jpg
    dog/*.jpg
```
**Tip:** You can use Kaggle's *Dogs vs Cats* dataset and manually split into `train/` and `val/` (e.g., 80/20).

### 1.4 Train model
```bash
python src/train.py --data_dir /path/to/DATA_DIR \
  --out_dir artifacts --epochs 3 --batch_size 32
```
Outputs: `artifacts/model.pt` and `artifacts/class_indices.json`.

### 1.5 Run API locally
```bash
uvicorn app.main:app --reload --port 8000
```
Open: `http://127.0.0.1:8000/docs` → try `/predict`.

---

## 2) FastAPI App (`app/main.py`)
```python
# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io, json, torch
import torchvision.transforms as T
from torchvision import models

app = FastAPI(title="Cats vs Dogs API", version="1.0")

device = torch.device("cpu")
MODEL_PATH = "artifacts/model.pt"
CLASS_PATH = "artifacts/class_indices.json"

# Preprocess to match training
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = None
idx_to_class = None


def build_model(num_classes: int):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = m.fc.in_features
    m.fc = torch.nn.Linear(in_features, num_classes)
    return m


@app.on_event("startup")
def load_model():
    global model, idx_to_class
    try:
        with open(CLASS_PATH, "r") as f:
            classes = json.load(f)  # list like ["cat","dog"]
        idx_to_class = {i: c for i, c in enumerate(classes)}
        model_local = build_model(num_classes=len(classes))
        state = torch.load(MODEL_PATH, map_location=device)
        model_local.load_state_dict(state)
        model_local.eval()
        model = model_local.to(device)
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG image.")
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_class = idx_to_class[pred_idx]
        confidence = float(probs[pred_idx])
        return JSONResponse({"prediction": pred_class, "confidence": confidence})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 3) Training Script (`src/train.py`)
```python
# src/train.py
import argparse, json, os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def make_datasets(data_dir: str, val_ratio: float = 0.2):
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    if val_dir.exists():
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
        classes = train_ds.classes
    else:
        full = datasets.ImageFolder(train_dir, transform=train_tf)
        classes = full.classes
        n_val = int(len(full) * val_ratio)
        n_train = len(full) - n_val
        train_ds, val_ds = random_split(full, [n_train, n_val])
        # Override val transform
        val_ds.dataset.transform = eval_tf
    return train_ds, val_ds, classes


def build_model(num_classes: int):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Path with train/ and optional val/")
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, classes = make_datasets(args.data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc, best_state = va_acc, model.state_dict()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(best_state or model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    with open(os.path.join(args.out_dir, "class_indices.json"), "w") as f:
        json.dump(list(classes), f)
    print(f"Saved model + classes to {args.out_dir}")

if __name__ == "__main__":
    main()
```

---

## 4) Requirements (`requirements.txt`)
```txt
fastapi==0.112.2
uvicorn[standard]==0.30.6
pillow==10.4.0
numpy==1.26.4
pydantic==2.9.0
python-multipart==0.0.9
pytest==8.3.2
httpx==0.27.0
# Install torch/torchvision separately per README (CPU wheels)
```

---

## 5) Docker

### 5.1 Ignore files (`.dockerignore`)
```gitignore
.venv
__pycache__
*.pyc
*.pyo
*.pyd
.git
.gitignore
.data
*.ipynb
*.DS_Store
```

### 5.2 Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app

# System deps for Pillow/torch vision ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy app code + artifacts (ensure you trained first!)
COPY app ./app
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6) Tests (`tests/test_api.py`)
```python
from fastapi.testclient import TestClient
from app.main import app

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
```

Run locally:
```bash
pytest -q
```

---

## 7) CI/CD via GitHub Actions (`.github/workflows/cicd.yml`)
This workflow builds and pushes your Docker image to **GitHub Container Registry (GHCR)** and (optionally) deploys to a VM over SSH.

**Before enabling:**
1. In your repo **Settings → Actions → General**, allow workflows.
2. In **Settings → Packages**, ensure you can publish to GHCR.
3. Create repo **Secrets**:
   - `EC2_HOST`, `EC2_USER`, `EC2_KEY` (private key for SSH)
   - `GHCR_PAT` (a Personal Access Token with `read:packages` for the VM pull step)

On your VM (once), install Docker and login to GHCR using `GHCR_PAT`.

```yaml
name: CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  IMAGE_NAME: cats-dogs-api

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.repository_owner }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:latest

  deploy:
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: SSH remote deploy
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USER }}
          key: ${{ secrets.EC2_KEY }}
          script: |
            sudo docker login ghcr.io -u ${{ github.repository_owner }} -p ${{ secrets.GHCR_PAT }}
            sudo docker pull ghcr.io/${{ github.repository_owner }}/cats-dogs-api:latest
            sudo docker rm -f catsdogs || true
            sudo docker run -d --name catsdogs --restart always -p 80:8000 ghcr.io/${{ github.repository_owner }}/cats-dogs-api:latest
```

---

## 8) Cloud VM Setup (AWS EC2 Free Tier)
1. Launch **Ubuntu 22.04 t2.micro**.
2. **Security Group**: open ports **22 (SSH)** and **80 (HTTP)**.
3. SSH in:
   ```bash
   ssh -i path/to/key.pem ubuntu@YOUR_PUBLIC_IP
   ```
4. Install Docker:
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   newgrp docker
   ```
5. (Optional) Test locally on VM:
   ```bash
   docker run --rm -p 80:8000 -v $(pwd)/artifacts:/app/artifacts ghcr.io/OWNER/cats-dogs-api:latest
   ```

> Tip: If your model is large later, mount `artifacts/` as a volume and keep images slim.

---

## 9) API Usage
- Swagger UI: `http://SERVER_IP_OR_DOMAIN/docs`
- `POST /predict` with `multipart/form-data` field `file` containing an image.
- Response:
```json
{"prediction": "cat", "confidence": 0.997}
```

---

## 10) Next Steps / Extensions
- Add **Streamlit** client app that calls the deployed API.
- Add unit tests for preprocessing and a sample image.
- Add **prometheus-fastapi-instrumentator** for basic metrics.
- Switch to **nginx reverse proxy** + TLS (Let's Encrypt) for production polish.

---

## 11) Troubleshooting
- **Torch install issues locally** → use the CPU index URL shown above.
- **Model not found** → ensure you ran training and copied `artifacts/` before building the Docker image.
- **Container runs but 404** → check port mapping; in Dockerfile we expose 8000; map `-p 80:8000` on the VM.
- **GHCR pull auth fails** → create `GHCR_PAT` with `read:packages` and use it in the deploy step.

---

## 12) Minimal Helpers (`src/utils.py`)
```python
# src/utils.py
from pathlib import Path

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)
```

