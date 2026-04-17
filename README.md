# Items Management App

A full-stack web application for managing items with a FastAPI backend and vanilla JavaScript frontend, containerized with Docker.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Browser       │    │   Docker        │
│                 │    │   Container     │
│  Frontend       │◄──►│                 │
│  (HTML/CSS/JS)  │    │  Backend        │
│                 │    │  (FastAPI)      │
└─────────────────┘    │                 │
                       │  Static Files   │
                       │  Served         │
                       └─────────────────┘
```

- **Frontend**: Vanilla HTML, CSS, and JavaScript served as static files
- **Backend**: FastAPI with CORS enabled for browser requests
- **Containerization**: Docker with docker-compose for easy deployment

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
