from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Pydantic model
class Item(BaseModel):
    name: str
    description: Optional[str] = None

# In-memory storage
items_db: dict[int, dict] = {}
next_id: int = 1

# Endpoints
@app.get("/items")
def get_items():
    return list(items_db.values())

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]

@app.post("/items", status_code=201)
def create_item(item: Item):
    global next_id
    new_item = {"id": next_id, "name": item.name, "description": item.description}
    items_db[next_id] = new_item
    next_id += 1
    return new_item

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    updated_item = {"id": item_id, "name": item.name, "description": item.description}
    items_db[item_id] = updated_item
    return updated_item

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_id]
    return {"message": "Item deleted"}

# Run with: uvicorn main:app --reload