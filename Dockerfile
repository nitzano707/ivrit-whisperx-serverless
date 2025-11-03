# ==========================================================
# WhisperX + FasterWhisper (Light Build for GitHub Codespaces)
# ==========================================================

FROM python:3.10-slim

# התקנת תלויות מערכת בסיסיות בלבד
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# העתקת הדרישות
COPY requirements.txt /app/requirements.txt

# התקנת חבילות Python קלות
RUN pip install --upgrade pip && \
    pip install torch==2.3.0+cpu torchaudio==2.3.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install git+https://github.com/m-bain/whisperX.git && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# העתקת קבצי האפליקציה
COPY app.py handler.py /app/

ENV WHISPER_MODEL=small
ENV PYTHONUNBUFFERED=1

CMD ["python3", "handler.py"]
