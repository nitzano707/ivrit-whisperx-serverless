# ==========================================================
# IVRIT-AI (faster-whisper) + WhisperX diarization
# RunPod Serverless ready | CPU build (אפשר לעבור ל-GPU ע"י שינוי device)
# ==========================================================

FROM python:3.10-slim

# ספריות מערכת הכרחיות
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# התקנת Torch CPU חד-פעמית (לשלוט בגרסה וביעד)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0+cpu torchaudio==2.3.0+cpu \
      -f https://download.pytorch.org/whl/cpu/torch_stable.html

# התקנת תלויות פייתון נוספות
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && rm -rf ~/.cache/pip

# קבצי האפליקציה
COPY app.py handler.py /app/

# משתני סביבה
ENV PYTHONUNBUFFERED=1
ENV WHISPERX_DISABLE_HF_AUTH=1
# אופציונלי להרצה מקומית/ענן – לא לשים ערכים קשיחים!
ENV RUNPOD_ENDPOINT_ID=""
ENV RUNPOD_API_KEY=""

# נקודת הכניסה ל-RunPod Serverless
CMD ["python3", "handler.py"]
