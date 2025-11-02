# --- בסיס: PyTorch תואם CUDA 12.1 ---
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# סביבת עבודה
WORKDIR /app

# מערכת + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# קאש למודלים (יישאר בליירים)
ENV HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    PYTORCH_ENABLE_MPS_FALLBACK=1

# התקנות פייתון
COPY requirements.txt .
# התקנת torchaudio תואם CUDA 12.1 (לפני שאר החבילות)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
      torchaudio==2.3.0+cu121 && \
    pip install --no-cache-dir -r requirements.txt

# קוד האפליקציה
COPY app.py .

# --- הורדת מודלים בזמן build (ללא שריפת טוקן בליירים) ---
# נדרש docker buildx/buildkit והעברת סוד:  --secret id=HF_TOKEN,env=HF_TOKEN
# 1) Whisper "small" (faster-whisper) ייכנס לקאש
RUN python - <<'PY'\nfrom faster_whisper import WhisperModel\nWhisperModel('small', device='cpu')\nPY

# 2) pyannote diarization (דורש טוקן HF בזמן build כדי לעבור gating)
# הטוקן יועבר זמנית כ-secret בזמן build ולא יישמר בלייר
RUN --mount=type=secret,id=HF_TOKEN \
    bash -lc 'export HUGGING_FACE_HUB_TOKEN=$(cat /run/secrets/HF_TOKEN); \
      python - << "PY"\n\
from huggingface_hub import login\n\
import os\n\
tok=os.environ.get("HUGGING_FACE_HUB_TOKEN")\n\
if not tok:\n\
    raise SystemExit("HF token missing at build time")\n\
login(token=tok)\n\
from pyannote.audio import Pipeline\n\
Pipeline.from_pretrained("pyannote/speaker-diarization-3.0")\n\
PY'

EXPOSE 8000

# הרצה
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
