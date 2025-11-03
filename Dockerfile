# ==========================================================
# WhisperX + FasterWhisper (Lightweight Build for RunPod)
# ==========================================================

FROM python:3.10-slim

# התקנת ספריות מערכת בסיסיות
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# הגדרת תיקיית עבודה
WORKDIR /app

# התקנת חבילות Python
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install git+https://github.com/m-bain/whisperX.git && \
    pip install -r requirements.txt

# העתקת קבצי האפליקציה
COPY app.py handler.py /app/

# משתני סביבה
ENV WHISPER_MODEL=small
ENV PYTHONUNBUFFERED=1

# נקודת כניסה ל-RunPod
CMD ["python3", "handler.py"]
