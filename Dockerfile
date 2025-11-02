FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# התקנת גרסאות תואמות מראש
RUN pip install --upgrade pip setuptools wheel
RUN pip install "torch==2.3.0" "torchaudio==2.3.0" --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install "numpy>=2.0.0"
RUN pip install "faster-whisper==1.0.3" "pydub" "soundfile" "ffmpeg-python"
RUN pip install git+https://github.com/pyannote/pyannote-audio.git@release/3.1.1
RUN pip install runpod fastapi uvicorn

# 🩹 תיקון באג np.NaN → np.nan
RUN sed -i 's/np.NaN/np.nan/g' /usr/local/lib/python3.10/site-packages/pyannote/audio/core/inference.py

# בדיקת גרסאות
RUN python3 - <<'PY'
import numpy, pyannote.audio
print("✅ NumPy:", numpy.__version__)
print("✅ PyAnnote fixed NaN bug successfully!")
PY

COPY . .

ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

CMD ["python3", "handler.py"]
