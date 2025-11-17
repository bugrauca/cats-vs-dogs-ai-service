# Cats vs Dogs Image Classification API

A fully containerized and cloud deployed machine learning service that classifies images of cats and dogs. The system includes model training, a FastAPI inference service, Docker packaging, automated testing, and CI/CD deployment to Render.

---

## 1. Project Overview

This project demonstrates a complete production oriented workflow for deploying an AI model as a scalable web service. It is designed to showcase practical skills in model development, API design, MLOps, and cloud deployment.

Key features:

- Model training with PyTorch
- FastAPI based inference API
- Docker containerization
- Automated tests
- CI pipeline with GitHub Actions
- Continuous deployment to Render

---

## 2. Repository Structure

```
.
├── app/                 # FastAPI application code and endpoints
│   └── api.py           # FastAPI app instance and predict/health routes
├── artifacts/           # Saved trained model and class index files
├── dataset/             # Image data used for training and validation
│   └── raw/             # Original downloads / unprocessed images
│   └── train/           # Training images (organized by class)
│   └── val/             # Validation images (organized by class)
├── src/                 # Training and utility scripts
│   └── train.py         # Model training script
├── tests/               # Automated tests for API and code
│   └── test_api.py      # Basic API health/test client checks
├── Dockerfile           # Container build instructions
├── requirements.txt     # Python dependencies
└── README.md            # Project overview and usage
```

---

## 3. Model Training

Run the training script after preparing your dataset.

```
python src/train.py --data_dir DATA_DIR --out_dir artifacts --epochs 3
```

The training script:

- Loads images
- Trains a CNN
- Saves `model.pt` into `artifacts/`

---

## 4. Running the API Locally

Install dependencies:

```
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Run the API:

```
uvicorn app.api:app --reload --port 8000
```

Visit OpenAPI documentation:

```
http://127.0.0.1:8000/docs
```

---

## 5. Docker Build and Run

Build the container:

```
docker build -t catsdogs .
```

Run it:

```
docker run -p 8000:8000 catsdogs
```

Access the service at `/docs`.

---

## 6. Testing

Tests are executed with pytest:

```
pytest
```

The pipeline runs tests automatically on each push.

---

## 7. CI/CD Pipeline (GitHub Actions)

A workflow file is included under `.github/workflows/cicd.yml`.
The pipeline:

- Installs dependencies
- Executes tests
- Deploys to Render on success

Render secrets required:

- `RENDER_API_KEY`
- `RENDER_SERVICE_ID`

---

## 8. Deployment on Render

Render automatically builds and runs the Dockerfile. Deployment triggers on each push to the main branch.
Once deployed, the service is available at:

```
https://cats-vs-dogs-cppi.onrender.com
```

---

## 9. Prediction Endpoint Example

Call the API using curl:

```
curl -X POST \
  -F "file=@cat.jpg" \
  https://cats-vs-dogs-cppi.onrender.com/predict
```

Example response:

```
{"label": "cat", "confidence": 0.94}
```

---

## 10. Future Improvements

Suggested enhancements:

- Add a small frontend for image upload
- Expand testing for inference logic
- Add monitoring or logging
- Improve model architecture

---

## Environment setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add `.venv/` to `.gitignore` to avoid committing the environment.

---

## Model / artifacts

The `artifacts/` folder holds the trained model and metadata (for example `model.pt` and `class_indices.json`).
Model binaries can be large — use Git LFS or external storage for model files.

---

## Data format & preprocessing

Dataset layout (under `dataset/`) should follow PyTorch's ImageFolder convention: one folder per class under `train/` and `val/`.
Images are preprocessed to 224x224 (center-crop after resize) and normalized using ImageNet statistics.

---

## API endpoints

- GET `/health` — health check; response: `200 {"status":"ok"}`
- POST `/predict` — form file field `file` (image/png or image/jpeg). Returns JSON: `{"prediction":"cat","confidence":0.92}` or `400/500` errors.

Example (local):

```bash
curl -X POST -F "file=@cat.jpg" http://127.0.0.1:8000/predict
```

---

## Tests (local)

Run tests from the project root (activate `.venv` first):

```bash
python -m pytest -q
```

Tests use FastAPI's TestClient and don't require the server to be running.

---

## Git LFS & large files

Large binaries and virtual environment files should not be committed directly. Track model files with Git LFS:

```bash
git lfs install
git lfs track "artifacts/*"
git add .gitattributes
git commit -m "Track artifacts with Git LFS"
```

If large files were already committed, migrate them with `git lfs migrate import --include="artifacts/**"` or remove them from history using tools like BFG or `git filter-repo`.

---

## Troubleshooting

- If startup fails check `artifacts/model.pt` and `artifacts/class_indices.json` exist and are readable.
- Pushes failing due to large files: use Git LFS or remove large blobs from history.
- Run with more logs: `uvicorn app.api:app --reload --log-level debug`.

---

## Contributing

Contributions welcome — open issues or PRs.
