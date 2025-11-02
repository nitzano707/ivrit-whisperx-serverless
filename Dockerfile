FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- שלב התקנות עם הקפאה מוחלטת ----
RUN pip install --upgrade pip setuptools wheel

# מתקינים קודם numpy ו-torch בגירסאות תואמות
RUN pip install "numpy==1.26.4" "torch==2.3.0" "torchaudio==2.3.0" --extra-index-url https://download.pytorch.org/whl/cu121

# מתקינים את pyannote.audio ישירות ממקור GitHub עם תלותים מתוקנים
RUN pip install git+https://github.com/pyannote/pyannote-audio.git@release/3.1.1 --no-deps

# כעת מתקינים את שאר הספריות
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install runpod

# הדפסת גרסאות לבדיקה
RUN python3 - <<'PY'
import numpy, torch, pyannote.audio
print("✅ NumPy:", numpy.__version__)
print("✅ Torch:", torch.__version__)
print("✅ PyAnnote:", pyannote.audio.__version__)
PY

COPY . .

ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

CMD ["python3", "handler.py"]
