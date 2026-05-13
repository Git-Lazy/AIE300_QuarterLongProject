from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
IRIS_LABELS = ["setosa", "versicolor", "virginica"]


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)


ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = SimpleClassifier(input_size=4, hidden_size=16, num_classes=3)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pth", weights_only=True))
    model.eval()
    ml["model"]  = model
    ml["scaler"] = joblib.load(MODEL_DIR / "scaler.pkl")
    yield
    ml.clear()


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": "model" in ml}


@app.post("/predict")
def predict(req: PredictionRequest):
    if len(req.features) != 4:
        raise HTTPException(status_code=422, detail="Expected exactly 4 features")
    scaled = ml["scaler"].transform([req.features])
    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = ml["model"](tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()
    return {
        "prediction": IRIS_LABELS[idx],
        "confidence": round(probs[idx].item(), 4),
        "model": "iris-classifier-v1",
    }
