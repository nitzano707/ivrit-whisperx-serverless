# --- שלב 1: בסיס CUDA עם PyTorch ---
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# --- שלב 2: הגדרת סביבת עבודה ---
WORKDIR /app

# --- שלב 3: התקנת ספריות ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- שלב 4: העתקת קובצי האפליקציה ---
COPY app.py .

# --- שלב 5: הורדת המודלים בזמן הבנייה ---
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu')" \
 && python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.0', use_auth_token='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')"

# --- שלב 6: פתיחת פורט ---
EXPOSE 8000

# --- שלב 7: הפעלת השרת ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
