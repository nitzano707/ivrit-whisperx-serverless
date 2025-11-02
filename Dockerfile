FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 🧩 התקנת תלויות בסיס
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git sed \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

# 🧠 התקנות עיקריות
RUN pip install --upgrade pip setuptools wheel
RUN pip install "torch==2.3.0" "torchaudio==2.3.0" --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install "numpy>=2.0.0"

# 📦 התקנת הדרישות (כולל faster-whisper וכו')
COPY requirements.txt .
RUN pip install -r requirements.txt

# 🔧 התקנת pyannote.audio – גם אם מה-PyPI, נתקן מיד אח"כ
RUN pip install --no-cache-dir pyannote.audio || pip install --no-cache-dir git+https://github.com/pyannote/pyannote-audio.git@release/4.0.1

# 🩹 תיקון אוטומטי של np.NaN → np.nan
RUN PYFILE=$(python3 -c "import inspect, pyannote.audio.core.inference as inf; print(inspect.getfile(inf))") \
 && echo '📄 Fixing np.NaN in' $PYFILE \
 && sed -i 's/np\.NaN/np.nan/g' $PYFILE \
 && echo '✅ Patch applied successfully!' \
 && grep -n "np\.nan" $PYFILE || true

# 🧪 בדיקה שהייבוא עובר
RUN python3 - <<'PY'
import numpy, inspect
print("✅ NumPy:", numpy.__version__)
import pyannote.audio
print("✅ pyannote.audio imported successfully")
PY

# העתקת קבצי האפליקציה
COPY . .

ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

CMD ["python3", "handler.py"]
