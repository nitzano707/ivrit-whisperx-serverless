# 🧩 CUDA בסיסית
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 🕓 תלויות בסיסיות
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 🧾 העתקת הדרישות
COPY requirements.txt .

# 🧠 התקנת הדרישות, אבל מכריחים גרסת NumPy תקינה
RUN pip install --upgrade pip \
 && pip install -r requirements.txt || true \
 && pip install --force-reinstall "numpy==1.26.4" \
 && pip check || true

# ✅ התקנת RunPod SDK
RUN pip install runpod

# 🧪 הדפסת גרסאות לאימות
RUN python3 -c "import numpy, torch; print('✅ NumPy:', numpy.__version__); print('✅ Torch:', torch.__version__)" || true

# 📦 העתקת כל קבצי האפליקציה
COPY . .

# 🔒 משתני סביבה
ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

# 🧠 הרצה — מתקין שוב NumPy בזמן עלייה, כדי למנוע דריסה
ENTRYPOINT ["sh", "-c", "pip install -q --force-reinstall numpy==1.26.4 && python3 handler.py"]
