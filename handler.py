#!/usr/bin/env python3
"""
RunPod Serverless Handler לתמלול וזיהוי דוברים עם WhisperX
"""
import runpod
import os, tempfile, base64, io, torch, logging
from faster_whisper import WhisperModel
from pydub import AudioSegment
import whisperx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

asr = None
dia = None

def load_models():
    global asr, dia
    if asr is None or dia is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Loading models on {device}")

        # Whisper
        model_size = os.getenv("WHISPER_MODEL", "small")
        asr = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")

        # WhisperX Diarization
        dia = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
        logger.info("✅ Models loaded successfully")

def process_audio_to_wav(audio_data: bytes) -> str:
    audio = AudioSegment.from_file(io.BytesIO(audio_data))
    audio = audio.set_frame_rate(16000).set_channels(1)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(tmp.name, format="wav")
    return tmp.name

def handler(job):
    """RunPod handler"""
    try:
        job_input = job["input"]
        audio_base64 = job_input.get("audio_base64")
        if not audio_base64:
            return {"error": "Missing audio_base64 in input"}

        language = job_input.get("language", "he")
        load_models()

        audio_data = base64.b64decode(audio_base64)
        wav_path = process_audio_to_wav(audio_data)

        # תמלול
        segments, info = asr.transcribe(
            wav_path, beam_size=5, language=language, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        transcription = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ]

        # זיהוי דוברים
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
            "language": language,
            "results": results,
            "speakers_count": len(set(r["speaker"] for r in results))
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
