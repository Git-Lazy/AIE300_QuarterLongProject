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

### Set environment variables

Make sure `OPENAI_API_KEY` is set before starting the app so the agent can call the OpenAI API.

```bash
set OPENAI_API_KEY=your_api_key_here
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
- `POST /agent` — Run the agent on a natural language task, using tools for search, creation, and RAG lookup
- `POST /agent/confirm` — Confirm or deny a pending destructive tool action when the agent requests approval

## Agent interface and safety guardrails

The frontend now includes a dedicated agent panel for natural language tasks, with explicit guardrails to keep tool use safe and transparent.

Guardrails:
- `POST /agent` is used for reasoning and tool selection only. This keeps the agent workflow separate from normal CRUD operations.
- `POST /agent/confirm` is required before executing destructive tools like `create_item` or `delete_item`. This prevents accidental or unexpected data changes.
- The UI shows a reasoning trace of the agent's tool calls and outputs. That transparency helps users understand what the agent did and why.

Why each one matters:
- explicit confirmation for destructive actions ensures the user stays in control and avoids unintended database changes.
- separating the agent flow from direct item endpoints reduces risk by limiting tool execution to a controlled path.
- trace output makes the agent's decisions visible, which improves trust and helps diagnose incorrect behavior.

## Screenshot

![App Screenshot](screenshot.png)

*Screenshot showing the items list and create form in the browser.*

## Development

- Frontend files are in `static/`
- Backend code is in `main.py`
- Use `docker-compose up --build` for development with live reload

## Prompt engineering and structured output (`/analyze`)

**System message used**: "You are a data analysis assistant. Analyze the provided content and respond with ONLY valid JSON in this exact format: { "categories": ["category1","category2"], "tags": ["tag1","tag2","tag3"], "sentiment": "positive" | "negative" | "neutral", "summary": "one sentence summary" } Do not include any text outside the JSON object." 

Why: This instructs the model to only emit a JSON object with the exact fields the server expects, reducing parsing failures and making downstream code simpler.

**Few-shot example included**:
- Input: "The new laptop is incredibly fast and the battery lasts all day. Best purchase this year."
- Output: {"categories": ["technology","review"], "tags": ["laptop","performance","battery"], "sentiment": "positive", "summary": "Highly positive review praising laptop speed and battery life."}

Why: A concrete example anchors the expected output shape and phrasing, improving reliability.

**Structured format expected**:
- `categories`: array of short category strings
- `tags`: array of short tag strings
- `sentiment`: one of `positive`, `negative`, `neutral`
- `summary`: a one-sentence English summary

**Failure handling**:
- Responses are parsed by extracting the first JSON object found in the model output and running `json.loads()`.
- If parsing fails, the backend retries once after appending a brief instruction asking the model to "Please respond with ONLY the JSON object and no surrounding text.".
- If retries fail, the endpoint returns a `4xx` or `5xx` HTTP error with a descriptive message. If a real secret was ever committed, remove it from history (see below).

This design balances prompt clarity (system + few-shot), deterministic decoding (low temperature), and pragmatic parsing/ retry logic to handle noisy model outputs.
