from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
import tempfile, os, torch, uvicorn

app = FastAPI(title="Transcription + Speaker Diarization API")

# --- טעינת מודלים מראש ---
print("🚀 Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

asr_model = WhisperModel("small", device=device)
dia_model = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=os.getenv("HF_TOKEN")
)
print("✅ Models loaded successfully!")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """מקבל קובץ אודיו -> מחזיר תמלול עם זיהוי דוברים"""
    try:
        # שמירת קובץ זמני
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio = AudioSegment.from_file(file.file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_in.name, format="wav")

        # שלב 1: תמלול
        print("🗣️ Running transcription...")
        segments, _ = asr_model.transcribe(temp_in.name, beam_size=5, language="he")
        transcript = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]

        # שלב 2: זיהוי דוברים
        print("👥 Running speaker diarization...")
        diarization = dia_model(temp_in.name)
        speakers = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # שלב 3: שילוב בין תמלול לדוברים
        final = []
        for seg in transcript:
            spk = next((s["speaker"] for s in speakers if s["start"] <= seg["start"] <= s["end"]), "unknown")
            final.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": spk,
                "text": seg["text"]
            })

        os.remove(temp_in.name)
        return {"status": "success", "results": final}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
