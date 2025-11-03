import runpod
import subprocess
import os
from app import process_audio

def download_youtube_audio(url):
    output_path = "/tmp/youtube.wav"
    subprocess.run([
        "yt-dlp", "-x", "--audio-format", "wav", "-o", output_path, url
    ], check=True)
    return output_path

def handler(event):
    try:
        input_data = event.get("input", {})
        audio_path = input_data.get("audio_path")
        youtube_url = input_data.get("youtube_url")

        if youtube_url:
            print("🔹 מוריד אודיו מיוטיוב...")
            audio_path = download_youtube_audio(youtube_url)

        if not audio_path or not os.path.exists(audio_path):
            return {"error": "לא נמצא קובץ אודיו או קישור תקין."}

        print("🔹 מתחיל עיבוד...")
        result = process_audio(audio_path)
        return {"status": "success", "segments": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
