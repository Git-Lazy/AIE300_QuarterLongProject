# Intelesense made the README
## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Server

```bash
python -m uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000` and the interactive docs at `http://127.0.0.1:8000/docs`.

## Available Endpoints

- `GET /items` — Return a list of all items
- `GET /items/{id}` — Return a single item by ID, or `404` if not found
- `POST /items` — Create a new item from JSON body (`201` on success)
- `PUT /items/{id}` — Update an existing item, or `404` if not found
- `DELETE /items/{id}` — Delete an item by ID, or `404` if not found
