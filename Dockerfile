# 🧩 שלב בסיסי – שימוש בתמונה רשמית עם תמיכת CUDA
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 🕓 עדכון מערכת והתקנת תלויות בסיסיות
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 📁 יצירת תיקיית העבודה
WORKDIR /app

# 🧾 העתקת קובץ הדרישות והתקנת ספריות
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ✅ התקנת RunPod SDK (לסביבת Serverless)
RUN pip install runpod

# 🧠 העתקת כל קבצי האפליקציה
COPY . .

# 🔒 משתני סביבה (ניתן לשנות ב-RunPod Dashboard)
ENV HF_TOKEN=""
ENV WHISPER_MODEL="small"

# 🧠 הורדת מודלים רק בעת ריצה (לא בשלב ה-build)
# זה מונע קובץ Docker כבד מדי.
# המודלים יורדו אוטומטית בקריאה הראשונה ל-handler.py

# ⚙️ פקודת ההפעלה של RunPod Serverless
CMD ["python3", "handler.py"]
