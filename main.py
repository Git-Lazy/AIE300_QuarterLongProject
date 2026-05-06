from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import torch
import torch.nn as nn
import joblib

# used the intele sense for the SQLAlchemy imports and ussages
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

BASE_DIR = Path(__file__).parent
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
    model.load_state_dict(torch.load(BASE_DIR / "model.pth", weights_only=True))
    model.eval()
    ml["model"]  = model
    ml["scaler"] = joblib.load(BASE_DIR / "scaler.pkl")
    yield
    ml.clear()

app = FastAPI(lifespan=lifespan)

# Database setup
DATABASE_URL = "sqlite:///./items.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class ItemModel(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class Item(BaseModel):
    name: str
    description: Optional[str] = None

# Prediction
class PredictionRequest(BaseModel):
    features: list[float]

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
    return {"prediction": IRIS_LABELS[idx], "confidence": round(probs[idx].item(), 4)}

# Endpoints
@app.get("/items")
def get_items(db: Session = Depends(get_db)):
    items = db.query(ItemModel).all()
    return [{"id": item.id, "name": item.name, "description": item.description} for item in items]

@app.get("/items/{item_id}")
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"id": item.id, "name": item.name, "description": item.description}

@app.post("/items", status_code=201)
def create_item(item: Item, db: Session = Depends(get_db)):
    db_item = ItemModel(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return {"id": db_item.id, "name": db_item.name, "description": db_item.description}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item, db: Session = Depends(get_db)):
    db_item = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    db_item.name = item.name
    db_item.description = item.description
    db.commit()
    db.refresh(db_item)
    return {"id": db_item.id, "name": db_item.name, "description": db_item.description}

@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(db_item)
    db.commit()
    return {"message": "Item deleted"}

# Mount static files after API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Run with: .\venv\Scripts\Activate.ps1
# python iris_classifier.py
