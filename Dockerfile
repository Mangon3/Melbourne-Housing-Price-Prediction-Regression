FROM python:3.11-slim

WORKDIR /app

# hadolint ignore=DL3008,DL3005
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install PyTorch CPU wheels from trusted index with pinned versions, then project deps
# tvdatafeed has no PyPI binary; must install from source via git.
RUN pip install --no-cache-dir --only-binary :all: torch==2.12.0+cpu torchvision==0.27.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --only-binary :all: -r requirements.txt \
    && pip install --upgrade --no-cache-dir --no-build-isolation git+https://github.com/rongardF/tvdatafeed.git \
    && pip install --upgrade --no-cache-dir pip setuptools wheel==0.46.2 jaraco.context==6.1.0

COPY . .
RUN pip install --no-cache-dir --only-binary :all: -e .

EXPOSE 7860
CMD ["uvicorn", "src.api.index:app", "--host", "0.0.0.0", "--port", "7860"]