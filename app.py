import os
import io
import base64
import json
import tempfile
import subprocess
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydub import AudioSegment
from faster_whisper import WhisperModel
import whisperx

# ----------------------------------------------------------
# הגדרות כלליות + device (GPU אם RUNPOD_GPU_COUNT קיים)
# ----------------------------------------------------------
DEVICE = "cuda" if os.environ.get("RUNPOD_GPU_COUNT") else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"  # מאיץ GPU, חוסך CPU

# טוענים מודלים פעם אחת (Warm)
_model_transcribe = None
_model_diar = None

def load_models():
    global _model_transcribe, _model_diar
    if _model_transcribe is None:
        _model_transcribe = WhisperModel(
            "ivrit-ai/faster-whisper-v2-d4",
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
    if _model_diar is None:
        # אין צורך ב-token (אתה כבר הגדרת DISABLE_HF_AUTH)
        _model_diar = whisperx.DiarizationPipeline(
            use_auth_token=None,
            device=DEVICE
        )

def ensure_wav_mono16k(input_path: str) -> str:
    """
    ממיר כל קובץ אודיו/וידאו לפורמט wav מונו 16k לצמצום עומס ולתמיכה מיטבית.
    """
    out_path = os.path.join(tempfile.gettempdir(), "audio_mono16k.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def extract_audio_from_video_if_needed(path: str) -> str:
    """
    אם הקובץ הוא וידאו – נפיק ממנו אודיו; אם כבר אודיו – נחזיר כמות שהוא.
    """
    lower = path.lower()
    if lower.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
        return ensure_wav_mono16k(path)
    return path

def split_audio(path: str,
                chunk_length_ms: int = 60_000,
                overlap_ms: int = 500) -> List[str]:
    """
    פיצול אודיו למקטעים עם חפיפה קלה לשמירת רצף.
    """
    audio = AudioSegment.from_file(path)
    chunks = []
    step = max(1, chunk_length_ms - overlap_ms)
    for start in range(0, len(audio), step):
        end = min(len(audio), start + chunk_length_ms)
        if end - start <= 0:
            break
        chunk = audio[start:end]
        chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{start}_{end}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        if end == len(audio):
            break
    return chunks

def transcribe_chunk(chunk_path: str) -> List[Dict]:
    """
    תמלול מקטע יחיד עם faster-whisper.
    """
    segments = []
    # שימוש באופציות מהירות (beam_size=1, VAD)
    for seg in _model_transcribe.transcribe(
        chunk_path,
        beam_size=1,
        vad_filter=True
    )[0]:
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })
    return segments

def diarize_chunk(chunk_path: str) -> List[Dict]:
    """
    זיהוי דוברים למקטע יחיד עם WhisperX.
    """
    audio = whisperx.load_audio(chunk_path)
    diar = _model_diar(audio)
    out = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        out.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": spk
        })
    return out

def best_overlap_speaker(t, diar_tracks) -> str:
    """
    מאתר את הדובר בעל החפיפה הטובה ביותר עבור תת-משפט מתומלל.
    """
    best_label, best_overlap = "SPEAKER_UNKNOWN", 0.0
    ts, te = t["start"], t["end"]
    for s in diar_tracks:
        ss, se = s["start"], s["end"]
        start = max(ts, ss)
        end = min(te, se)
        ov = max(0.0, end - start)
        if ov > best_overlap:
            best_overlap = ov
            best_label = s["speaker"]
    return best_label

def merge_transcript_diar(transcription: List[Dict], diarization: List[Dict]) -> List[Dict]:
    """
    מיזוג: לכל מקטע טקסט – משייכים דובר לפי חפיפת הזמן הגדולה ביותר.
    """
    merged = []
    for t in transcription:
        spk = best_overlap_speaker(t, diarization)
        merged.append({**t, "speaker": spk})
    return merged

def parallel_process_chunk(chunk_path: str) -> List[Dict]:
    """
    מריץ תמלול + דוברים במקביל על מקטע, מחזיר תוצאה מאוחדת לאותו מקטע.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_trans = pool.submit(transcribe_chunk, chunk_path)
        f_diar = pool.submit(diarize_chunk, chunk_path)
        trans = f_trans.result()
        diar = f_diar.result()
    return merge_transcript_diar(trans, diar)

def clean_temp_files(paths: List[str]):
    for p in paths:
        try:
            os.remove(p)
        except Exception:
            pass

def process_audio(input_path: str) -> List[Dict]:
    """
    תהליך מלא:
    - חילוץ אודיו מוידאו (אם צריך) → המרה ל-wav 16k mono
    - פיצול למקטעים (עם חפיפה)
    - הרצה מקבילית של תמלול+דוברים לכל מקטע
    - איחוד תוצאות ורשימת מקטעים סופית
    """
    load_models()
    # הכנה
    prepared = extract_audio_from_video_if_needed(input_path)
    wav = ensure_wav_mono16k(prepared) if prepared != input_path else ensure_wav_mono16k(input_path)

    # קביעה: אם קצר מ-2 דקות – ללא פיצול
    audio_len_ms = len(AudioSegment.from_file(wav))
    if audio_len_ms <= 120_000:
        chunks = [wav]
    else:
        chunks = split_audio(wav, chunk_length_ms=60_000, overlap_ms=500)

    results_all = []
    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(parallel_process_chunk, c) for c in chunks]
        for f in as_completed(futures):
            results_all.extend(f.result())

    # ניקוי זמניים (בלי למחוק wav אם הוא שימש כ-chunk יחיד)
    if len(chunks) > 1:
        clean_temp_files(chunks)
    if os.path.exists(wav):
        try:
            os.remove(wav)
        except Exception:
            pass

    # מיון לפי זמן התחלה
    results_all.sort(key=lambda x: (x["start"], x["end"]))
    return results_all

# -----------------------
# עוזרים ל-handler
# -----------------------
def save_b64_to_wav(data_url_b64: str) -> str:
    """
    מקבל data URL (data:audio/...;base64,XXXX) וכותב לקובץ זמני wav/mp3.
    נחזיר נתיב קובץ פיזי לעיבוד.
    """
    # פירוק prefix
    if "," in data_url_b64:
        _, b64 = data_url_b64.split(",", 1)
    else:
        b64 = data_url_b64
    raw = base64.b64decode(b64)
    tmp_in = os.path.join(tempfile.gettempdir(), "upload_input.bin")
    with open(tmp_in, "wb") as f:
        f.write(raw)
    # המרה ל-wav מועבר ב-process_audio
    return tmp_in

def download_youtube_audio(url: str) -> str:
    """
    הורדת אודיו מ-YouTube בעזרת yt-dlp, חיסכון לרוחב: WAV 16k mono.
    """
    out_wav = os.path.join(tempfile.gettempdir(), "youtube_16k.wav")
    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav", "-o", out_wav, url
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav
