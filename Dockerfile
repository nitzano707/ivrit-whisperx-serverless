FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 🕓 התקנת תלויות בסיס
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 🧾 התקנת חבילות מרכזיות עם תאימות ל-NumPy 2.x
RUN pip install --upgrade pip setuptools wheel
RUN pip install "torch==2.3.0" "torchaudio==2.3.0" --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install "numpy>=2.0.0"

# 🧠 התקנת הספריות שלך
COPY requirements.txt .
RUN pip install -r requirements.txt

# 🧩 התקנת גרסה עדכנית של pyannote.audio שתומכת ב-NumPy 2.x
RUN pip install "pyannote.audio>=3.3.1"

# ✅ התקנת RunPod SDK
RUN pip install runpod

# 🧪 בדיקת תאימות
RUN python3 - <<'PY'
import numpy, pyannote.audio
print("✅ NumPy:", numpy.__version__)
print("✅ PyAnnote:", pyannote.audio.__version__)
PY

# 📦 העתקת קבצי האפליקציה
COPY . .

# 🔒 משתני סביבה
ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

# ⚙️ הפעלה רגילה
CMD ["python3", "handler.py"]
