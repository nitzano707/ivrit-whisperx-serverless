FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# ffmpeg לאודיו
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir "numpy<2.0"

COPY app.py .

# (אופציונלי) טעינת מודלים בזמן build כדי לקצר זמן אתחול — יגדיל את האימג'
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu')" \
#  && python -c "from pyannote.audio import Pipeline; import os; Pipeline.from_pretrained('pyannote/speaker-diarization-3.0', use_auth_token=os.getenv('HF_TOKEN',''))"

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
