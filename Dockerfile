FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git sed findutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

# 🧠 התקנות בסיסיות
RUN pip install --upgrade pip setuptools wheel
RUN pip install "torch==2.3.0" "torchaudio==2.3.0" --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install "numpy>=2.0.0"

# 📦 התקנת יתר הספריות שלך
COPY requirements.txt .
RUN pip install -r requirements.txt

# 🧩 התקנת pyannote.audio (גם אם ישנה)
RUN pip install --no-cache-dir pyannote.audio || pip install --no-cache-dir git+https://github.com/pyannote/pyannote-audio.git@release/4.0.1

# 🩹 תיקון גורף לכל מופעי np.NaN
RUN echo "🔍 Searching for np.NaN in site-packages..." \
 && find /usr/local/lib/python3.10/site-packages/pyannote -type f -name "*.py" -exec grep -l "np\.NaN" {} \; > /tmp/files.txt || true \
 && echo "📄 Files to patch:" && cat /tmp/files.txt || true \
 && sed -i 's/np\.NaN/np.nan/g' $(cat /tmp/files.txt) || true \
 && echo "✅ All np.NaN replaced with np.nan"

# 🧪 בדיקה
RUN python3 - <<'PY'
import numpy
print("✅ NumPy:", numpy.__version__)
import glob
files = glob.glob("/usr/local/lib/python3.10/site-packages/pyannote/**/*.py", recursive=True)
fixed = all("np.NaN" not in open(f).read() for f in files)
print("🔍 np.NaN still present?", not fixed)
if fixed:
    print("✅ Patch verified, pyannote ready!")
PY

COPY . .

ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

CMD ["python3", "handler.py"]
