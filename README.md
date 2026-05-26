---
title: Stock Agent
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Stock Agent API

A FastAPI-powered financial analysis platform using LangGraph, Google Gemini, and machine learning for comprehensive stock analysis.

## Features

- **RAG-Enhanced News Analysis**: ChromaDB-powered retrieval for financial news
- **ML-Based Price Prediction**: GRU neural network for technical analysis
- **LangGraph Agent**: Multi-tool orchestration for dynamic analysis
- **Redis Caching**: 24-hour cache for API responses
- **Rate Limit Handling**: Automatic retry with exponential backoff

## Live Demo

[https://huggingface.co/spaces/mangonnnn/stock-agent](https://huggingface.co/spaces/mangonnnn/stock-agent)

## API Endpoint

### POST `/analyze`

**Request:**

```json
{
  "symbol": "AAPL",
  "timeframe_days": 10,
  "query": "Analyze Apple's AI strategy" // optional
}
```

**Response:**

```json
{
  "symbol": "AAPL",
  "macro_analysis": "...",
  "micro_analysis": {...},
  "final_report": "..."
}
```

## Local Development

### Prerequisites

- Python 3.11+
- Docker (optional)

### Setup

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/stock-agent.git
cd stock-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run Locally

```bash
uvicorn src.api.index:app --reload --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker-compose up
```

## Tech Stack

- **Framework**: FastAPI
- **Agent**: LangGraph
- **LLM**: Google Gemini 2.5 Flash
- **Vector DB**: ChromaDB
- **Cache**: Redis
- **ML**: PyTorch + scikit-learn
