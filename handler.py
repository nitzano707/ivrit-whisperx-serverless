import os
import runpod
from app import process_audio, save_b64_to_wav, download_youtube_audio

# אופציונלי: קריאה למשתני סביבה שנשמרו ב-RunPod (אם תרצה להשתמש בהם)
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")

def handler(event):
    """
    קלט צפוי (JSON):
    {
      "input": {
        "youtube_url": "...",         # אופציונלי
        "audio_b64": "data:audio/wav;base64,...."  # אופציונלי
      }
    }
    חייב להיות youtube_url או audio_b64 (אחד מהם לפחות).
    """
    try:
        data = (event or {}).get("input", {}) or {}
        youtube_url = data.get("youtube_url", "").strip()
        audio_b64 = data.get("audio_b64", "")

        if not youtube_url and not audio_b64:
            return {"status": "error", "message": "נא לספק youtube_url או audio_b64"}

        if youtube_url:
            in_path = download_youtube_audio(youtube_url)
        else:
            in_path = save_b64_to_wav(audio_b64)

        # עיבוד מלא (תמלול + דוברים + מיזוג)
        segments = process_audio(in_path)

        # ניקוי קובץ המקור המקומי
        try:
            if in_path and os.path.exists(in_path):
                os.remove(in_path)
        except Exception:
            pass

        return {"status": "success", "segments": segments}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
