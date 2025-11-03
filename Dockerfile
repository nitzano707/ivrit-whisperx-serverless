# ==========================================================
# Combined IVRIT-AI Transcription + WhisperX Diarization
# RunPod Serverless ready
# ==========================================================

FROM python:3.10-slim

# התקנת ספריות מערכת
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# התקנת חבילות Python
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# העתקת קבצי האפליקציה
COPY app.py handler.py /app/

ENV PYTHONUNBUFFERED=1
ENV WHISPERX_DISABLE_HF_AUTH=1

CMD ["python3", "handler.py"]
