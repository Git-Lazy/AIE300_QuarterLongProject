# READ ME created completely by intelesense
# Items Management App

A full-stack web application for managing items with a FastAPI backend and vanilla JavaScript frontend, containerized with Docker.

## Architecture

Three containers, orchestrated by docker-compose:

```
                                  ┌──────────────────────────┐
                                  │  /health checks          │
                                  │  (compose waits for      │
                                  │   each service to pass)  │
                                  └──────────────────────────┘
                                              ▲
                                              │
┌──────────┐   HTTP   ┌──────────────┐   HTTP ┌────────────┐   HTTP   ┌──────────────────┐
│  Client  │ ───────► │  frontend    │ ─────► │   api      │ ───────► │  model-service   │
│ (browser │          │  (:8080)     │        │  (:8000)   │          │     (:8001)      │
│  curl…)  │ ◄─────── │  nginx       │ ◄───── │  FastAPI   │ ◄─────── │  FastAPI + torch │
└──────────┘          │  serves UI + │        │  /predict  │          │  loads model.pth │
                      │  proxies API │        │  proxies → │          │  + scaler.pkl    │
                      └──────────────┘        └────────────┘          └──────────────────┘

Request flow for a prediction:
  Browser  →  http://localhost:8080/predict           (frontend, nginx)
           →  http://api:8000/predict                  (proxied to api service)
           →  http://model-service:8001/predict       (forwarded by api)
           →  inference, response trickles back up the same path
```

The client never talks to `model-service` directly — only the `api` service can reach it (Docker compose network).

### Services

| Service         | Tech              | Host port | Purpose                                           |
| --------------- | ----------------- | --------- | ------------------------------------------------- |
| `frontend`      | nginx:alpine      | 8080      | Serves `static/index.html`; reverse-proxies API   |
| `api`           | FastAPI + uvicorn | 8000      | Items CRUD + `/predict` (forwards to model)       |
| `model-service` | FastAPI + PyTorch | 8001      | Loads the trained Iris model; exposes `/predict`  |

Each service has a `/health` endpoint and a docker-compose `healthcheck` so startup order is `model-service → api → frontend`.

## Quick Start with Docker

1. Ensure Docker is installed and running
2. Clone the repository
3. Run the application:

```bash
docker-compose up --build
```

4. Open `http://localhost:8000` in your browser

## Manual Setup (Alternative)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Server

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

## Screenshot

![App Screenshot](screenshot.png)

*Screenshot showing the items list and create form in the browser.*

## Development

- Frontend files are in `static/`
- Backend code is in `main.py`
- Use `docker-compose up --build` for development with live reload
