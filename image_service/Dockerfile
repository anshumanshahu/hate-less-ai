FROM python:3.12-slim

# Tesseract + Hindi language pack
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV HF_HOME=/models PYTHONUNBUFFERED=1

# CPU-only torch (GPU wale packages ke bina chhota image banega)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CLIP model build ke time download karke image me bake kar do
RUN python -c "from transformers import CLIPModel, CLIPProcessor; m='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(m); CLIPProcessor.from_pretrained(m)"

COPY app.py ocr.py image_model.pkl ./

# Runtime pe internet ki zaroorat nahi
ENV HF_HUB_OFFLINE=1
EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]