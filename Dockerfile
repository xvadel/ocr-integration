# Dockerfile for RegOCR: Egyptian Commercial Register OCR Service
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

# Install system dependencies: Tesseract 5 with Arabic & English training data, Poppler, OpenCV deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml README.md fields.yaml LICENSES.md ./
COPY src/ ./src/

# Install python dependencies
RUN uv pip install --system -e . --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/

EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "regocr.api:app", "--host", "0.0.0.0", "--port", "8000"]
