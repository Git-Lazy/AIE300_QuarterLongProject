import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
import requests
from dotenv import load_dotenv
from openai import OpenAI
import json
import time
import re
import ast

# used the intele sense for the SQLAlchemy imports and ussages
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")

app = FastAPI()

# Serve static files from /static and return index.html from /
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

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


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str


class ToolApproval(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    approved: bool


class AgentRequest(BaseModel):
    task: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class AgentConfirmRequest(BaseModel):
    task: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    pending_tool: Dict[str, Any]
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    approved: bool


class AgentResponse(BaseModel):
    result: Optional[str] = None
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    status: str = "complete"
    pending_tool: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


TOOL_SCHEMAS = [
    {
        "name": "search_items",
        "description": "Search items in the database by keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_item",
        "description": "Create a new item in the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Item name"},
                "description": {"type": "string", "description": "Item description"}
            },
            "required": ["name", "description"]
        }
    },
    {
        "name": "delete_item",
        "description": "Delete an item by ID from the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {"type": "integer", "description": "ID of the item to delete"}
            },
            "required": ["item_id"]
        }
    },
    {
        "name": "query_rag",
        "description": "Query the RAG system for a content answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question to ask the RAG system"}
            },
            "required": ["query"]
        }
    }
]

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


def get_items_from_db(query: str = "") -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        if query:
            items = db.query(ItemModel).filter(
                (ItemModel.name.ilike(f"%{query}%")) |
                (ItemModel.description.ilike(f"%{query}%"))
            ).all()
        else:
            items = db.query(ItemModel).all()
        return [{"id": item.id, "name": item.name, "description": item.description} for item in items]
    finally:
        db.close()


def create_item_in_db(name: str, description: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        db_item = ItemModel(name=name, description=description)
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        return {"id": db_item.id, "name": db_item.name, "description": db_item.description}
    finally:
        db.close()


def delete_item_in_db(item_id: int) -> str:
    db = SessionLocal()
    try:
        item = db.query(ItemModel).filter(ItemModel.id == item_id).first()
        if not item:
            return f"No item found with id {item_id}."
        db.delete(item)
        db.commit()
        return f"Deleted item {item_id}: {item.name}."
    finally:
        db.close()


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    matches = get_items_from_db(request.query)
    if not matches:
        return AskResponse(answer=f"No relevant items found for '{request.query}'.")

    answer_lines = [f"{item['name']}: {item['description']}" for item in matches[:5]]
    answer = f"Found {len(matches)} matching items. " + " ".join(answer_lines)
    return AskResponse(answer=answer)


def search_items_tool(query: str) -> str:
    matches = get_items_from_db(query)
    if not matches:
        return f"No items found for '{query}'."
    return json.dumps(matches)


def create_item_tool(name: str, description: str) -> str:
    created = create_item_in_db(name, description)
    return json.dumps(created)


def delete_item_tool(item_id: int) -> str:
    return delete_item_in_db(item_id)


def query_rag_tool(query: str) -> str:
    try:
        response = requests.post(f"{API_BASE}/ask", json={"query": query}, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", str(data))
    except requests.RequestException as exc:
        return f"RAG request failed: {exc}"


def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    if tool_name == "search_items":
        return search_items_tool(tool_args["query"])
    if tool_name == "create_item":
        return create_item_tool(tool_args["name"], tool_args["description"])
    if tool_name == "delete_item":
        return delete_item_tool(tool_args["item_id"])
    if tool_name == "query_rag":
        return query_rag_tool(tool_args["query"])
    raise ValueError(f"Unknown tool: {tool_name}")


def requires_confirmation(tool_name: str) -> bool:
    return tool_name in {"create_item", "delete_item"}


def parse_tool_call(message: Any) -> Tuple[str, Dict[str, Any]]:
    if getattr(message, "tool_calls", None):
        tool_call = message.tool_calls[0]
        arguments = tool_call.function.arguments
        name = tool_call.function.name
    elif getattr(message, "function_call", None):
        tool_call = message.function_call
        arguments = tool_call.arguments
        name = tool_call.name
    else:
        raise ValueError("No tool call found in the response message.")

    try:
        parsed_args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON arguments from model: {exc}\n{arguments}") from exc

    return name, parsed_args


def call_llm(messages: List[Dict[str, Any]]) -> Any:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_SCHEMAS,
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message


def run_agent(task: str, conversation_history: List[Dict[str, str]] = None, max_steps: int = 10):
    if conversation_history is None:
        conversation_history = []

    system_msg = {
        "role": "system",
        "content": (
            "You are an agent that can use tools to search the item database, "
            "create or delete items, and query the RAG system. "
            "Only call a tool when it is needed, and return a final answer when done."
        )
    }

    messages: List[Dict[str, Any]] = [system_msg] + conversation_history + [{"role": "user", "content": task}]
    steps: List[Dict[str, Any]] = []

    for iteration in range(max_steps):
        model_message = call_llm(messages)
        if not getattr(model_message, "tool_calls", None) and not getattr(model_message, "function_call", None):
            result = model_message.content or ""
            return {
                "status": "complete",
                "result": result,
                "steps": steps,
                "conversation_history": conversation_history + [
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": result}
                ]
            }

        tool_name, tool_args = parse_tool_call(model_message)
        if requires_confirmation(tool_name):
            return {
                "status": "needs_confirmation",
                "steps": steps,
                "conversation_history": conversation_history,
                "pending_tool": {
                    "name": tool_name,
                    "args": tool_args,
                    "description": next(tool["description"] for tool in TOOL_SCHEMAS if tool["name"] == tool_name)
                },
                "message": f"The agent wants to call {tool_name} with {tool_args}. Please approve to continue."
            }

        try:
            tool_output = execute_tool(tool_name, tool_args)
        except Exception as exc:
            tool_output = f"Tool failed: {exc}"

        steps.append({"tool": tool_name, "input": tool_args, "output": tool_output})
        messages.append({"role": "tool", "name": tool_name, "content": tool_output})

    return {
        "status": "complete",
        "result": "Step limit reached.",
        "steps": steps,
        "conversation_history": conversation_history
    }


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(request: AgentRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured. Set OPENAI_API_KEY in environment.")

    response = run_agent(request.task, request.conversation_history)
    return AgentResponse(**response)


@app.post("/agent/confirm", response_model=AgentResponse)
def agent_confirm(request: AgentConfirmRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured. Set OPENAI_API_KEY in environment.")

    if not request.approved:
        return AgentResponse(
            result="Action cancelled by user.",
            steps=request.steps,
            conversation_history=request.conversation_history,
            status="complete",
            message="The destructive action was not approved."
        )

    pending_tool = request.pending_tool
    tool_name = pending_tool["name"]
    tool_args = pending_tool["args"]

    try:
        tool_output = execute_tool(tool_name, tool_args)
    except Exception as exc:
        tool_output = f"Tool failed: {exc}"

    steps = request.steps + [{"tool": tool_name, "input": tool_args, "output": tool_output}]
    updated_history = request.conversation_history + [
        {"role": "user", "content": request.task},
        {"role": "assistant", "content": f"Calling {tool_name} with {tool_args}."},
        {"role": "tool", "name": tool_name, "content": tool_output}
    ]
    messages = [
        {"role": "system", "content": (
            "You are an agent that can use tools to search the item database, "
            "create or delete items, and query the RAG system. "
            "Only call a tool when it is needed, and return a final answer when done."
        )}
    ] + updated_history

    for _ in range(5):
        model_message = call_llm(messages)
        if not getattr(model_message, "tool_calls", None) and not getattr(model_message, "function_call", None):
            result = model_message.content or ""
            return AgentResponse(
                result=result,
                steps=steps,
                conversation_history=updated_history + [{"role": "assistant", "content": result}],
                status="complete"
            )

        next_tool_name, next_tool_args = parse_tool_call(model_message)
        if requires_confirmation(next_tool_name):
            return AgentResponse(
                steps=steps,
                conversation_history=updated_history,
                status="needs_confirmation",
                pending_tool={
                    "name": next_tool_name,
                    "args": next_tool_args,
                    "description": next(tool["description"] for tool in TOOL_SCHEMAS if tool["name"] == next_tool_name)
                },
                message=f"The agent wants to call {next_tool_name} with {next_tool_args}. Please approve to continue."
            )

        try:
            next_tool_output = execute_tool(next_tool_name, next_tool_args)
        except Exception as exc:
            next_tool_output = f"Tool failed: {exc}"

        steps.append({"tool": next_tool_name, "input": next_tool_args, "output": next_tool_output})
        messages.append({"role": "tool", "name": next_tool_name, "content": next_tool_output})

    return AgentResponse(
        result="Step limit reached.",
        steps=steps,
        conversation_history=updated_history,
        status="complete"
    )
