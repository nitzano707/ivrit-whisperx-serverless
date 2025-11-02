FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# התקנת ספריות בסיס
RUN pip install --upgrade pip setuptools wheel
RUN pip install "torch==2.3.0" "torchaudio==2.3.0" --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install "numpy>=2.0.0"

# התקנת שאר התלויות שלך
COPY requirements.txt .
RUN pip install -r requirements.txt

# הסרת pyannote הישנה והתקנת גרסה חדשה תואמת ל-NumPy 2.x
RUN pip uninstall -y pyannote.audio || true
RUN pip install --no-cache-dir --force-reinstall "pyannote.audio>=3.3.2"

# אימות גרסאות
RUN python3 - <<'PY'
import numpy, pyannote.audio
print("✅ NumPy:", numpy.__version__)
print("✅ PyAnnote:", pyannote.audio.__version__)
PY

COPY . .

ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

CMD ["python3", "handler.py"]
