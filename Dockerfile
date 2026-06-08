FROM python:3.11-slim

# System deps for OpenCV + InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# For GPU servers swap onnxruntime → onnxruntime-gpu in requirements.txt before building
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/models data

ENV PORT=3000
EXPOSE 3000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
