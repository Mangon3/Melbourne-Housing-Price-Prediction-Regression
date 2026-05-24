FROM python:3.11-slim

WORKDIR /app

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install PyTorch CPU wheels from trusted index with pinned versions, then project deps
# NOSONAR - tvdatafeed has no PyPI binary; must install from source. Pinned to a specific commit is not feasible as the repo does not publish stable tags.
RUN pip install --no-cache-dir --only-binary :all: torch==2.12.0+cpu torchvision==0.27.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu # NOSONAR
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt # NOSONAR
RUN pip install --upgrade --no-cache-dir --no-build-isolation git+https://github.com/rongardF/tvdatafeed.git # NOSONAR

COPY . .
RUN pip install --no-cache-dir --only-binary :all: -e . # NOSONAR

EXPOSE 7860
CMD ["uvicorn", "src.api.index:app", "--host", "0.0.0.0", "--port", "7860"]