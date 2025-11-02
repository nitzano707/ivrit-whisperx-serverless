import os, io, tempfile, json, time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch
import logging

# הגדרת לוגים
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# קריאת משתנים מסביבה
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("⚠️ HF_TOKEN לא מוגדר - pyannote עלול להיכשל")

# יצירת אפליקציית FastAPI
app = FastAPI(
    title="תמלול וזיהוי דוברים",
    version="1.0.0"
)

# הגדרת CORS פתוח
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# משתנים גלובליים למודלים
asr = None
dia = None

def load_models():
    """טעינת המודלים פעם אחת"""
    global asr, dia
    
    if asr is None or dia is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 טוען מודלים על: {device}")
        
        # טעינת Whisper
        model_size = os.getenv("WHISPER_MODEL", "small")
        logger.info(f"טוען Whisper מודל: {model_size}")
        asr = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
        
        # טעינת Pyannote
        logger.info("טוען Pyannote...")
        dia = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=HF_TOKEN
        )
        if device == "cuda":
            dia.to(torch.device("cuda"))
        
        logger.info("✅ המודלים נטענו בהצלחה!")

# טעינת המודלים בעת הפעלה
@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/", response_class=HTMLResponse)
async def home():
    """דף HTML להעלאת קבצים"""
    return HTML_CONTENT

@app.get("/health")
async def health():
    """בדיקת תקינות השרת"""
    return {
        "status": "healthy",
        "models_loaded": asr is not None and dia is not None,
        "cuda_available": torch.cuda.is_available()
    }

def to_wav_16k_mono(upload: UploadFile) -> str:
    """המרת קובץ אודיו ל-WAV 16kHz מונו"""
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "קובץ ריק")
    
    # המרה ל-WAV
    audio = AudioSegment.from_file(io.BytesIO(data))
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # שמירה כקובץ זמני
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(tmp.name, format="wav")
    return tmp.name

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = "he"
):
    """תמלול קובץ אודיו עם זיהוי דוברים"""
    
    # בדיקת גודל קובץ
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"הקובץ גדול מדי. מקסימום: {MAX_FILE_SIZE/1024/1024}MB")
    
    # החזרת הקובץ לתחילתו
    file.file = io.BytesIO(file_content)
    
    wav_path = None
    try:
        start_time = time.time()
        logger.info(f"מתחיל תמלול קובץ: {file.filename}")
        
        # וודא שהמודלים טעונים
        if asr is None or dia is None:
            load_models()
        
        # המרת הקובץ
        wav_path = to_wav_16k_mono(file)
        
        # תמלול עם Whisper
        logger.info("מתחיל תמלול...")
        segments, info = asr.transcribe(
            wav_path, 
            beam_size=5, 
            language=language,
            vad_filter=True,  # סינון רעשים
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        transcription = []
        for segment in segments:
            transcription.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
        
        # זיהוי דוברים עם Pyannote
        logger.info("מזהה דוברים...")
        diarization = dia(wav_path)
        
        speakers = []
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            speakers.append({
                "start": round(float(turn.start), 2),
                "end": round(float(turn.end), 2),
                "speaker": speaker_label
            })
        
        # שילוב תמלול עם דוברים
        logger.info("משלב תוצאות...")
        final_results = []
        for trans_seg in transcription:
            best_speaker = "SPEAKER_UNKNOWN"
            best_overlap = 0.0
            
            for spk_seg in speakers:
                # חישוב חפיפה
                overlap_start = max(trans_seg["start"], spk_seg["start"])
                overlap_end = min(trans_seg["end"], spk_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_speaker = spk_seg["speaker"]
            
            final_results.append({
                "start": trans_seg["start"],
                "end": trans_seg["end"],
                "text": trans_seg["text"],
                "speaker": best_speaker
            })
        
        # חישוב זמן עיבוד
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"✅ התמלול הושלם תוך {processing_time} שניות")
        
        return {
            "status": "success",
            "filename": file.filename,
            "language": language,
            "processing_time": processing_time,
            "results": final_results,
            "speakers_count": len(set(s["speaker"] for s in final_results))
        }
        
    except Exception as e:
        logger.error(f"שגיאה בתמלול: {str(e)}")
        raise HTTPException(500, f"שגיאת שרת: {str(e)}")
    
    finally:
        # ניקוי קובץ זמני
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except:
                pass

# HTML פשוט לבדיקה
HTML_CONTENT = """
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>תמלול וזיהוי דוברים</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
        .segment { margin: 10px 0; padding: 10px; background: #f5f5f5; }
        .speaker { font-weight: bold; color: #007bff; }
        .loading { display: none; color: #666; }
    </style>
</head>
<body>
    <h1>🎤 תמלול אוטומטי עם זיהוי דוברים</h1>
    
    <div class="upload-area">
        <input type="file" id="fileInput" accept="audio/*">
        <button onclick="transcribe()">התחל תמלול</button>
        <div class="loading" id="loading">מעבד... אנא המתן...</div>
    </div>
    
    <div id="results"></div>
    
    <script>
        async function transcribe() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('אנא בחר קובץ אודיו');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayResults(data);
                } else {
                    alert('שגיאה בתמלול');
                }
            } catch (error) {
                alert('שגיאה: ' + error);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            let html = '<h2>תוצאות התמלול:</h2>';
            html += '<p>מספר דוברים: ' + data.speakers_count + '</p>';
            html += '<p>זמן עיבוד: ' + data.processing_time + ' שניות</p>';
            
            data.results.forEach(segment => {
                html += '<div class="segment">';
                html += '<span class="speaker">' + segment.speaker + ':</span> ';
                html += segment.text;
                html += ' <small>(' + segment.start + '-' + segment.end + 's)</small>';
                html += '</div>';
            });
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
"""
