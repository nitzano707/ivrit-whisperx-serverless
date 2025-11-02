import os
import io
import json
import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# אודיו
from pydub import AudioSegment

# ASR
import torch
from faster_whisper import WhisperModel

# Diarization (אופציונלי אם יש טוקן)
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.core import Segment


# -----------------------------
# הגדרות כלליות
# -----------------------------
APP_TITLE = "Transcription + Speaker Diarization API"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()  # ← לשים ברנפוד / דוקר כ־ENV
ASR_MODEL = os.getenv("ASR_MODEL", "small")  # small / medium / large-v3 / או מסלול מקומי
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"  # מהיר ב-GPU, חסכוני ב-CPU

app = FastAPI(title=APP_TITLE)

# CORS – מאפשר דפדפן/דומיינים לגשת
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # אפשר לצמצם לדומיינים קונקרטיים
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# -----------------------------
# טעינת מודלים בעת עליית השרת
# -----------------------------
print("🚀 Loading ASR model:", ASR_MODEL, "| device:", DEVICE, "| compute:", COMPUTE_TYPE)
asr_model = WhisperModel(ASR_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)

dia_model: Optional[PyannotePipeline] = None
if HF_TOKEN:
    try:
        print("👥 Loading diarization model (pyannote)...")
        dia_model = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=HF_TOKEN
        )
        # אפשרויות שיפור ביצועים
        if DEVICE == "cuda":
            dia_model.to(torch.device("cuda"))
        print("✅ Diarization model loaded.")
    except Exception as e:
        print("⚠️ Failed to load diarization model:", e)
        dia_model = None
else:
    print("ℹ️ No HF_TOKEN provided: diarization will be disabled.")


# -----------------------------
# עזר: שמירת קובץ והמרה ל-WAV 16kHz mono
# -----------------------------
def save_as_wav_16k_mono(upload: UploadFile) -> str:
    """
    שומר קובץ זמני כ-WAV בקצב 16kHz ומונו — מתאים ל-ASR ולדיאריזציה.
    """
    suffix = os.path.splitext(upload.filename or "")[-1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw_tmp:
        raw_bytes = upload.file.read()
        raw_tmp.write(raw_bytes)
        raw_path = raw_tmp.name

    # המרה ל-WAV 16kHz mono
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out_path = out_tmp.name
    out_tmp.close()

    audio = AudioSegment.from_file(raw_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(out_path, format="wav")

    try:
        os.remove(raw_path)
    except Exception:
        pass

    return out_path


# -----------------------------
# עזר: שילוב דוברים עם תמלול
# -----------------------------
def merge_speakers_with_transcript(
    transcript: List[dict],
    speaker_turns: List[dict]
) -> List[dict]:
    """
    עבור כל סגמנט תמלול — מוצא את הדובר המכסה את נקודת האמצע של הסגמנט.
    אם אין, מנסה לחפיפה >50%; ואם אין — "לא זוהה".
    """
    def pick_speaker(seg_start: float, seg_end: float) -> str:
        mid = 0.5 * (seg_start + seg_end)
        for s in speaker_turns:
            if s["start"] <= mid <= s["end"]:
                return s["speaker"]
        # חפיפה חלקית משמעותית
        best = ("לא זוהה", 0.0)
        for s in speaker_turns:
            inter = max(0.0, min(seg_end, s["end"]) - max(seg_start, s["start"]))
            dur = max(1e-9, seg_end - seg_start)
            frac = inter / dur
            if frac > best[1]:
                best = (s["speaker"], frac)
        return best[0] if best[1] > 0.5 else "לא זוהה"

    merged = []
    for seg in transcript:
        spk = pick_speaker(seg["start"], seg["end"])
        merged.append({**seg, "speaker": spk})
    return merged


# -----------------------------
# סכמות קלט/פלט
# -----------------------------
class URLInput(BaseModel):
    file_url: str


# -----------------------------
# בריאות
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "asr_model": ASR_MODEL,
        "diarization": bool(dia_model),
        "title": APP_TITLE,
    }


# -----------------------------
# תמלול + דוברים מקובץ שהועלה (multipart/form-data)
# -----------------------------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    מקבל קובץ אודיו (MP3/WAV/…)
    מחזיר: [{"start": float, "end": float, "speaker": str, "text": str}, ...]
    """
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    # שמירת קובץ והמרה ל-16kHz מונו
    tmp_wav = save_as_wav_16k_mono(file)

    try:
        # --- ASR ---
        print("🗣️ Transcribing…")
        segments, info = asr_model.transcribe(
            tmp_wav,
            language="he",
            beam_size=5,
            vad_filter=True
        )
        transcript = []
        for s in segments:
            transcript.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip()
            })

        # --- Diarization (אם נטען) ---
        speakers = []
        if dia_model is not None:
            print("👥 Diarizing…")
            diar = dia_model(tmp_wav)
            for turn, _, spk in diar.itertracks(yield_label=True):
                speakers.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(spk)
                })
        else:
            # אין מודל, נסמן כ"לא זוהה"
            speakers = []

        # --- שילוב ---
        if speakers:
            final = merge_speakers_with_transcript(transcript, speakers)
        else:
            final = [{**seg, "speaker": "לא זוהה"} for seg in transcript]

        return {"status": "success", "results": final}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass


# -----------------------------
# תמלול + דוברים מ-URL (JSON) — אופציונלי
# -----------------------------
@app.post("/transcribe-url")
async def transcribe_url(body: URLInput):
    """
    מקבל JSON: {"file_url": "https://.../audio.wav"}
    מוריד, ממיר, מתמלל ומחזיר תוצאות בדומה ל-/transcribe
    """
    import subprocess

    url = body.file_url
    if not url:
        raise HTTPException(status_code=400, detail="file_url is required")

    # הורדה לקובץ זמני
    raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    raw_tmp.close()
    raw_path = raw_tmp.name
    try:
        # curl עם -L להפניות
        cmd = ["curl", "-L", url, "-o", raw_path]
        subprocess.check_call(cmd)

        # המרה ל-16kHz מונו
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        audio = AudioSegment.from_file(raw_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")

        # תמלול
        segments, info = asr_model.transcribe(
            wav_path,
            language="he",
            beam_size=5,
            vad_filter=True
        )
        transcript = []
        for s in segments:
            transcript.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip()
            })

        # דיאריזציה (אם יש)
        speakers = []
        if dia_model is not None:
            diar = dia_model(wav_path)
            for turn, _, spk in diar.itertracks(yield_label=True):
                speakers.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(spk)
                })

        if speakers:
            final = merge_speakers_with_transcript(transcript, speakers)
        else:
            final = [{**seg, "speaker": "לא זוהה"} for seg in transcript]

        return {"status": "success", "results": final}

    except subprocess.CalledProcessError:
        raise HTTPException(status_code=400, detail="failed to download file_url")
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        for p in (raw_path, locals().get("wav_path", None)):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# -----------------------------
# פיתוח מקומי (לא חובה ברנפוד)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
