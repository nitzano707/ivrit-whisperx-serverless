# 🧩 שלב בסיסי – תמונה עם תמיכת CUDA לצורך Torch
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 🕓 עדכון מערכת והתקנת תלויות בסיסיות
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 📁 תיקיית העבודה
WORKDIR /app

# 🧾 העתקת הדרישות והתקנת ספריות
COPY requirements.txt .
RUN pip install --upgrade pip
# מתקין את כל הספריות + מכריח NumPy להישאר בגרסה תואמת
RUN pip install -r requirements.txt \
 && pip install -U numpy==1.26.4 \
 && echo "✅ Installed NumPy version:" && python3 -c "import numpy; print(numpy.__version__)"

# ✅ התקנת RunPod SDK (לסביבת Serverless)
RUN pip install runpod

# 🧠 העתקת כל קבצי האפליקציה
COPY . .

# 🔒 משתני סביבה (ניתן להגדיר מחדש בלוח הבקרה של RunPod)
ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

# 🧠 המודלים יורדו רק בזמן ריצה, לא בשלב הבנייה
# זה מקטין משמעותית את גודל התמונה.

# ⚙️ הפקודה הראשית – Serverless Handler
CMD ["python3", "handler.py"]
