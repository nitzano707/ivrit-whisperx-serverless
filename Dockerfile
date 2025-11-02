FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/models/hf TRANSFORMERS_CACHE=/models/hf

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torchaudio==2.3.0+cu121 && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Prefetch מהיר ל-ASR (לא דורש טוקן)
RUN python - <<'PY'\nfrom faster_whisper import WhisperModel\nWhisperModel('small', device='cpu')\nPY

CMD ["python", "-u", "handler.py"]
