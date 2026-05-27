import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import requests
from dotenv import load_dotenv
from openai import OpenAI
import json
import time
import re

# used the intele sense for the SQLAlchemy imports and ussages
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# When running under docker-compose this resolves to the model-service container.
# When running locally with uvicorn, it falls back to localhost:8001.
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")

app = FastAPI()

# Load environment variables from .env when present
load_dotenv()

# Initialize OpenAI client (will read key from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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
    try:
        response = requests.post(
            f"{MODEL_SERVICE_URL}/predict",
            json={"features": req.features},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Model service unavailable: {exc}")
    return response.json()

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

@app.get("/health")
def health():
    return {"status": "healthy", "model_service": MODEL_SERVICE_URL}


# --- Chat endpoint ---
class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    conversation_history: List[Dict[str, str]]


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured. Set OPENAI_API_KEY in environment.")

    # System message tailored to app domain
    system_msg = {"role": "system", "content": "You are a helpful assistant for an items manager app. Be concise and helpful."}

    # Build messages: system + history + user
    messages = [system_msg]
    # Validate that history entries look like messages
    for entry in request.conversation_history:
        if not (isinstance(entry, dict) and "role" in entry and "content" in entry):
            raise HTTPException(status_code=400, detail="Invalid conversation_history format. Expected list of {role, content} dicts.")
        messages.append(entry)

    messages.append({"role": "user", "content": request.message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )

        reply = response.choices[0].message.content

        updated_history = request.conversation_history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": reply}
        ]

        return ChatResponse(reply=reply, conversation_history=updated_history)

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")

    @app.post("/analyze")
    def analyze(request: BaseModel = None):
        # Define Pydantic-style request parsing manually to avoid circular redeclaration
        class AnalyzeRequest(BaseModel):
            content: str

        if request is None:
            raise HTTPException(status_code=400, detail="Missing body")

        # Parse incoming request into the AnalyzeRequest model
        try:
            if isinstance(request, AnalyzeRequest):
                payload = request
            else:
                payload = AnalyzeRequest(**request.dict()) if hasattr(request, 'dict') else AnalyzeRequest(**request)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

        if client is None:
            raise HTTPException(status_code=500, detail="OpenAI client not configured. Set OPENAI_API_KEY in environment.")

        system_prompt = (
            "You are a data analysis assistant. Analyze the provided content "
            "and respond with ONLY valid JSON in this exact format:\n"
            "{\n  \"categories\": [\"category1\", \"category2\"],\n  \"tags\": [\"tag1\", \"tag2\", \"tag3\"],\n  \"sentiment\": \"positive\" | \"negative\" | \"neutral\",\n  \"summary\": \"one sentence summary\"\n}\n"
            "Do not include any text outside the JSON object."
        )

        few_shot = (
            "Example:\n"
            "Input: \"The new laptop is incredibly fast and the battery lasts all day. Best purchase this year.\"\n"
            "Output: {\"categories\": [\"technology\", \"review\"], \"tags\": [\"laptop\", \"performance\", \"battery\"], \"sentiment\": \"positive\", \"summary\": \"Highly positive review praising laptop speed and battery life.\"}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": few_shot + "\n\nNow analyze this:\n" + payload.content}
        ]

        # Low temperature for consistent structured output
        attempts = 2
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=512,
                    temperature=0.2
                )

                raw = response.choices[0].message.content

                # Try to extract the JSON object from the response
                start = raw.find('{')
                end = raw.rfind('}')
                if start == -1 or end == -1 or end < start:
                    raise json.JSONDecodeError('No JSON object found', raw, 0)

                json_text = raw[start:end+1]
                result = json.loads(json_text)

                # Validate required fields
                required = ["categories", "tags", "sentiment", "summary"]
                for field in required:
                    if field not in result:
                        raise ValueError(f"Missing field: {field}")

                return result

            except json.JSONDecodeError:
                last_err = HTTPException(status_code=422, detail="LLM returned invalid JSON. Retrying...")
                # on retry, add explicit instruction to return only JSON
                messages.append({"role": "user", "content": "Please respond with ONLY the JSON object and no surrounding text."})
                time.sleep(0.5)
                continue
            except Exception as e:
                last_err = e
                break

        # If we exit loop with error
        if isinstance(last_err, HTTPException):
            raise last_err
        raise HTTPException(status_code=500, detail=f"Analyze failed: {last_err}")

    # Run with: .\venv\Scripts\Activate.ps1
    # python iris_classifier.py
