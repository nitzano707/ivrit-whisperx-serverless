import os, io, tempfile, json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch

HF_TOKEN = os.getenv("HF_TOKEN")  # שים ב־RunPod ENV
if not HF_TOKEN:
    print("⚠️  HF_TOKEN לא מוגדר — pyannote ייכשל אם המודל gated/private")

app = FastAPI(title="Transcribe + Speakers")

# CORS פתוח כדי לאפשר קריאה מדף HTML סטטי
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# טען מודלים פעם אחת
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading models on: {device}")
asr = WhisperModel("small", device=device)  # אפשר d→"medium"/"large-v3" אם תרצה
dia = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=HF_TOKEN
)
print("✅ Models loaded.")

@app.get("/health")
def health():
    return {"ok": True}

def to_wav_16k_mono(upload: UploadFile) -> str:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "קובץ ריק")
    audio = AudioSegment.from_file(io.BytesIO(data))
    audio = audio.set_frame_rate(16000).set_channels(1)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(tmp.name, format="wav")
    return tmp.name

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "he"):
    try:
        wav_path = to_wav_16k_mono(file)

        # --- תמלול ---
        segments, info = asr.transcribe(
            wav_path, beam_size=5, language=language
        )
        trs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]

        # --- דוברים ---
        diar = dia(wav_path)
        spk = [
            {"start": float(turn.start), "end": float(turn.end), "speaker": label}
            for turn, _, label in diar.itertracks(yield_label=True)
        ]

        # --- שילוב: משייך לכל סגמנט הדובר עם חפיפה מירבית ---
        final = []
        for t in trs:
            best = "UNKNOWN"; best_ovlp = 0.0
            for d in spk:
                ovlp = max(0.0, min(t["end"], d["end"]) - max(t["start"], d["start"]))
                if ovlp > best_ovlp:
                    best_ovlp = ovlp
                    best = d["speaker"]
            final.append({**t, "speaker": best})

        os.remove(wav_path)
        return {"status": "success", "results": final, "lang": language}

    except Exception as e:
        raise HTTPException(500, f"Server error: {e}")
