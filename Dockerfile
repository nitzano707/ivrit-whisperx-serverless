# ==========================================================
# WhisperX Diarization only - for RunPod Serverless
# ==========================================================

FROM python:3.10-slim

# התקנת תלויות מערכת בסיסיות בלבד
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# התקנת ספריות
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# העתקת קבצי האפליקציה
COPY app.py handler.py /app/

# משתני סביבה
ENV PYTHONUNBUFFERED=1
ENV WHISPERX_DISABLE_HF_AUTH=1

CMD ["python3", "handler.py"]
