# --- שלב 1: בסיס קליל עם CUDA ---
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# --- שלב 2: סביבת עבודה ---
WORKDIR /app

# --- שלב 3: התקנת תלויות ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- שלב 4: העתקת קבצי הקוד ---
COPY app.py .

# --- שלב 5: הורדת המודלים מראש ---
# Whisper small
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu')" \
 && python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.0', use_auth_token='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')"

# --- שלב 6: פתיחת הפורט ל-RunPod ---
EXPOSE 8000

# --- שלב 7: הפעלת האפליקציה ---
CMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
