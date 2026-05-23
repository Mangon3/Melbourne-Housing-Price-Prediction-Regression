FROM python:3.11-slim

WORKDIR /app

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# NOSONAR - tvdatafeed must be installed from source (no PyPI binary available)
RUN pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git

RUN pip install --no-cache-dir --only-binary :all: torch==2.12.0+cpu torchvision==0.27.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY . .
# NOSONAR - editable install of local project, source is trusted
RUN pip install --no-cache-dir -e .

EXPOSE 7860
CMD ["uvicorn", "src.api.index:app", "--host", "0.0.0.0", "--port", "7860"]