FROM runpod/base:0.4.0-cuda11.8.0

# הגדרת משתני סביבה
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/tmp

# עדכון והתקנת תלויות מערכת
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# יצירת תיקיית עבודה
WORKDIR /app

# העתקת requirements והתקנה
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# הורדת מודלים מראש (אופציונלי - יאיץ את הפעלה ראשונה)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', download_root='/tmp/whisper')"

# העתקת קבצי האפליקציה
COPY app.py .
COPY handler.py .

# פורט לבדיקות מקומיות
EXPOSE 8000

# הגדרת נקודת כניסה ל-RunPod
CMD ["python3", "handler.py"]
