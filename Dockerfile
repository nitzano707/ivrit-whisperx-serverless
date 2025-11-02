FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

# 📦 התקנת ספריות עיקריות
RUN pip install --upgrade pip setuptools wheel
RUN pip install "torch==2.3.0" "torchaudio==2.3.0" --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install "numpy>=2.0.0"

# העתקת הדרישות והתקנתן (כולל faster-whisper, fastapi וכו')
COPY requirements.txt .
RUN pip install -r requirements.txt

# התקנת pyannote.audio מה־GitHub או PyPI
RUN pip install --no-cache-dir git+https://github.com/pyannote/pyannote-audio.git@release/3.1.1 || pip install pyannote.audio

# 🩹 תיקון קובץ ה־inference.py במקרה שעדיין יש בו np.NaN
RUN PYFILE=$(python3 -c "import inspect, pyannote.audio.core.inference as inf; print(inspect.getfile(inf))") \
 && echo "📄 Fixing $PYFILE" \
 && sed -i 's/np\.NaN/np.nan/g' $PYFILE \
 && grep -n "np\.nan" $PYFILE

# 🧠 אימות
RUN python3 - <<'PY'
import numpy, pyannote.audio, inspect
from pathlib import Path
p = Path(inspect.getfile(pyannote.audio.core.inference))
print("✅ NumPy:", numpy.__version__)
print("✅ inference.py path:", p)
print("🔍 contains np.NaN?", "np.NaN" in p.read_text())
PY

COPY . .

ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

CMD ["python3", "handler.py"]
