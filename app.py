import os, io, tempfile, json, time, logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydub import AudioSegment
from faster_whisper import WhisperModel
import whisperx
import torch

# הגדרת לוגים
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# יצירת אפליקציית FastAPI
app = FastAPI(title="תמלול וזיהוי דוברים - WhisperX", version="2.0.0")

# הגדרת CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# משתנים גלובליים
asr = None
dia = None

def load_models():
    """טעינת המודלים פעם אחת"""
    global asr, dia
    if asr is None or dia is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 טוען מודלים על: {device}")

        # תמלול עם Whisper
        model_size = os.getenv("WHISPER_MODEL", "small")
        logger.info(f"🗣️ טוען Whisper {model_size}")
        asr = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")

        # זיהוי דוברים עם WhisperX
        logger.info("🎙️ טוען מודל זיהוי דוברים של WhisperX...")
        dia = whisperx.DiarizationPipeline(use_auth_token=None, device=device)

        logger.info("✅ כל המודלים נטענו בהצלחה!")

@app.on_event("startup")
async def startup_event():
    load_models()

def to_wav_16k_mono(upload: UploadFile) -> str:
    """המרת קובץ לאודיו WAV 16kHz מונו"""
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "קובץ ריק")
    audio = AudioSegment.from_file(io.BytesIO(data))
    audio = audio.set_frame_rate(16000).set_channels(1)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(tmp.name, format="wav")
    return tmp.name

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h1>WhisperX Server פעיל ✅</h1>"

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "he"):
    """תמלול קובץ עם זיהוי דוברים"""
    wav_path = None
    try:
        start_time = time.time()
        logger.info(f"🎧 התחלת תמלול: {file.filename}")
        load_models()

        # המרת קובץ
        wav_path = to_wav_16k_mono(file)

        # תמלול Whisper
        segments, info = asr.transcribe(
            wav_path, beam_size=5, language=language, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        transcription = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ]

        # זיהוי דוברים עם WhisperX
        logger.info("🎙️ מזהה דוברים עם WhisperX...")
        audio = whisperx.load_audio(wav_path)
        diarization_result = dia(audio)
        diarized_segments = whisperx.assign_word_speakers(diarization_result, {"segments": transcription})

        results = [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"],
                "speaker": seg.get("speaker", "SPEAKER_UNKNOWN")
            }
            for seg in diarized_segments["segments"]
        ]

        return {
            "status": "success",
            "filename": file.filename,
            "language": language,
            "processing_time": round(time.time() - start_time, 2),
            "results": results,
            "speakers_count": len(set(r["speaker"] for r in results))
        }

    except Exception as e:
        logger.error(f"שגיאה: {str(e)}")
        raise HTTPException(500, f"שגיאת שרת: {str(e)}")
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
