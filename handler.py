import os, io, base64, tempfile, json, subprocess
from typing import Optional
import runpod
from pydub import AudioSegment
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as PyannotePipeline

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
ASR_MODEL = os.getenv("ASR_MODEL", "small")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# --- טוען מודלים פעם אחת ---
asr_model = WhisperModel(ASR_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)

dia_model: Optional[PyannotePipeline] = None
if HF_TOKEN:
    try:
        dia_model = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=HF_TOKEN
        )
        if DEVICE == "cuda":
            dia_model.to(torch.device("cuda"))
    except Exception as e:
        print("pyannote disabled:", e)

def to_wav_16k(file_bytes: bytes) -> str:
    tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".bin"); tmp_raw.write(file_bytes); tmp_raw.close()
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav"); tmp_wav.close()
    audio = AudioSegment.from_file(tmp_raw.name)
    audio.set_frame_rate(16000).set_channels(1).export(tmp_wav.name, format="wav")
    os.remove(tmp_raw.name)
    return tmp_wav.name

def download_to_bytes(url: str) -> bytes:
    tmp = tempfile.NamedTemporaryFile(delete=False); tmp.close()
    subprocess.check_call(["curl", "-L", url, "-o", tmp.name])
    with open(tmp.name, "rb") as f: b = f.read()
    os.remove(tmp.name)
    return b

def merge_speakers(transcript, spk_segments):
    if not spk_segments:
        return [{**t, "speaker": "לא זוהה"} for t in transcript]
    def pick(s, e):
        mid = 0.5*(s+e)
        for seg in spk_segments:
            if seg["start"] <= mid <= seg["end"]:
                return seg["speaker"]
        # fallback: חפיפה מירבית
        best = ("לא זוהה", 0.0)
        for seg in spk_segments:
            inter = max(0.0, min(e, seg["end"]) - max(s, seg["start"]))
            frac = inter / max(e - s, 1e-6)
            if frac > best[1]: best = (seg["speaker"], frac)
        return best[0] if best[1] > 0.5 else "לא זוהה"

    return [{**t, "speaker": pick(t["start"], t["end"])} for t in transcript]

def handler(job):
    """
    input:
      file_b64: base64 של הקובץ (מומלץ לקבצים קצרים/בינוניים)
      או
      file_url: קישור ציבורי לקובץ (מומלץ לגדולים)
      language: ברירת מחדל "he"
    """
    data = job.get("input", {}) or {}
    lang = data.get("language", "he")

    try:
        if "file_b64" in data and data["file_b64"]:
            b64 = data["file_b64"]
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            audio_bytes = base64.b64decode(b64)
        elif "file_url" in data and data["file_url"]:
            audio_bytes = download_to_bytes(data["file_url"])
        else:
            return {"status": "error", "message": "Provide file_b64 or file_url"}

        wav_path = to_wav_16k(audio_bytes)

        # --- תמלול ---
        segments, _ = asr_model.transcribe(
            wav_path, language=lang, beam_size=5, vad_filter=True
        )
        transcript = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]

        # --- זיהוי דוברים ---
        spk_segments = []
        if dia_model is not None:
            diar = dia_model(wav_path)
            for turn, _, spk in diar.itertracks(yield_label=True):
                spk_segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)})

        result = merge_speakers(transcript, spk_segments)
        return {"status": "success", "results": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        try:
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

runpod.serverless.start({"handler": handler})
