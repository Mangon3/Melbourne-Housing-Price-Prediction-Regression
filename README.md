---
title: Stock Agent
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI Stock Agent

A comprehensive financial analysis platform consisting of a **Next.js Frontend** and a **FastAPI Backend**. It utilizes a LangGraph agent to orchestrate Google Gemini, a GRU machine learning model for price prediction, and a ChromaDB vector database for RAG (Retrieval-Augmented Generation) on financial news.

## Features

- **Frontend UI**: Sleek, modern chat interface deployed on Vercel.
- **RAG-Enhanced News Analysis**: ChromaDB-powered retrieval for financial news.
- **ML-Based Price Prediction**: GRU neural network for technical analysis.
- **LangGraph Agent**: Multi-tool orchestration for dynamic analysis.
- **DevSecOps Pipeline**: Fully automated Jenkins CI/CD with SonarCloud & Trivy security scanning.

## Live Demos

- **Frontend UI (Vercel)**: [https://stock-agent-frontend-three.vercel.app/](https://stock-agent-frontend-three.vercel.app/)
- **Backend API (Hugging Face)**: [https://huggingface.co/spaces/mangonnnn/stock-agent](https://huggingface.co/spaces/mangonnnn/stock-agent)

## Local Development (Docker)

The easiest way to run the entire backend infrastructure (FastAPI, Redis, Prometheus) locally is via Docker Compose.

### Prerequisites

- Docker & Docker Compose

### Setup & Run

```bash
# Clone the repository
git clone https://github.com/Mangon3/Stock-Agent.git
cd Stock-Agent

# Configure your environment variables
cp .env.example .env
# Edit .env and add your API keys (GOOGLE_API_KEY, FINNHUB_API_KEY)

# Build and start the infrastructure
docker-compose --project-directory . -f infra/docker-compose.yml up -d --build
```

_The API will be available at `http://localhost:7860/analyze`._

## Testing the Application

### 1. Local API Testing (Without Docker)

You can run the FastAPI server and the Python client directly from your terminal. Open two terminal windows:

**Terminal 1 (Start the Server):**

```bash
cd backend
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the API
uvicorn src.api.index:app --reload --port 8000
```

**Terminal 2 (Run the Client):**

```bash
cd backend
source .venv/bin/activate
export API_URL="http://localhost:8000/analyze"
python -m src.main
```

### 2. Online Testing (Vercel)

The frontend application is automatically deployed to Vercel upon every push to the `main` branch.
To test the web interface:

1. Visit your live Vercel URL.
2. Click the **Settings (Gear Icon)** in the top right.
3. Enter your **Google Gemini API Key**.
4. Type a query like `"Analyze Apple"` into the chat box to see the RAG agent stream the report in real-time.

## Tech Stack

- **Frontend**: Next.js, React, Tailwind CSS, Framer Motion
- **Backend**: Python, FastAPI
- **AI/ML**: LangGraph, Google Gemini 2.5, PyTorch, ChromaDB
- **Infrastructure**: Docker, Jenkins, Prometheus, Redis
