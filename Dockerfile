# 🧩 בסיס עם CUDA עבור Torch
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 🕓 עדכון מערכת והתקנת תלויות
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 📁 תיקיית עבודה
WORKDIR /app

# 🧾 התקנת דרישות עם כפייה של numpy ישנה
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt || true
# כפייה מלאה של גרסת NumPy
RUN pip install --force-reinstall "numpy==1.26.4"

# ✅ התקנת RunPod SDK
RUN pip install runpod

# הדפסת גרסת NumPy בזמן build כדי לוודא
RUN python3 -c "import numpy; print('✅ NumPy version in image:', numpy.__version__)"

# 🧠 העתקת קבצי האפליקציה
COPY . .

# 🔒 משתני סביבה
ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

# 🩹 כפיית NumPy גם בזמן ריצה (ליתר ביטחון)
ENTRYPOINT ["sh", "-c", "pip install -q --force-reinstall numpy==1.26.4 && python3 handler.py"]
