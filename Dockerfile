FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# תלויות בסיס
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1
WORKDIR /app

# התקנות בסיס יציבות לתאימות CUDA 12.1
RUN pip install --upgrade pip setuptools wheel
RUN pip install "torch==2.3.0" "torchaudio==2.3.0" --extra-index-url https://download.pytorch.org/whl/cu121
# NumPy 2.x כדי לוודא שאנחנו בתרחיש העדכני
RUN pip install "numpy>=2.0.0"

# התקנת שאר התלויות שלך (בלי pyannote.audio!)
COPY requirements.txt .
RUN sed -n '1,200p' requirements.txt
RUN pip install -r requirements.txt

# ניקוי כל שריד של pyannote/pyannote.audio
RUN pip uninstall -y pyannote.audio pyannote || true

# התקנת pyannote.audio ישירות מ־GitHub (גרסה עדכנית עם תיקוני NumPy 2.x)
# אם תרצה לנעול לגרסה מסוימת: החלף ל- @release/4.0.1 או ל- @3.3.2 אם קיים בענף ה-releases
RUN pip install --no-deps --no-cache-dir "git+https://github.com/pyannote/pyannote-audio.git@main"

# אימות: האם עדיין יש np.NaN בקובץ הבעייתי?
RUN python3 - <<'PY'
import inspect, pyannote.audio, numpy
import sys, os
from pathlib import Path
print("✅ NumPy:", numpy.__version__)
import pyannote.audio.core.inference as inf
p = Path(inspect.getfile(inf))
print("📄 inference.py path:", p)
text = p.read_text()
print("🔎 contains 'np.NaN'? ->", 'np.NaN' in text)
PY

# אם (מכל סיבה) יש np.NaN, מתקנים במקום ל-np.nan
RUN python3 - <<'PY'
import inspect, pyannote.audio.core.inference as inf
from pathlib import Path
p = Path(inspect.getfile(inf))
txt = p.read_text()
if "np.NaN" in txt:
    print("🩹 Patching np.NaN -> np.nan in", p)
    txt = txt.replace("np.NaN", "np.nan")
    p.write_text(txt)
else:
    print("✅ No patch needed.")
PY

# בדיקת סופית אחרי הפאץ'
RUN python3 - <<'PY'
import inspect, pyannote.audio.core.inference as inf
from pathlib import Path
p = Path(inspect.getfile(inf))
print("🔁 Re-check:", 'np.NaN' in p.read_text())
import pyannote.audio
print("✅ pyannote.audio imported OK. Version attribute may not exist on main; import succeeded = good.")
PY

# העתקת קוד האפליקציה
COPY . .

# משתני סביבה
ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

# הפעלה (RunPod serverless)
CMD ["python3", "handler.py"]
