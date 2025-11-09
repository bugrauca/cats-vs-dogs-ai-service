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
preprocess = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

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
