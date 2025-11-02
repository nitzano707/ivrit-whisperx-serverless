#!/usr/bin/env python3
"""
RunPod Serverless Handler לתמלול וזיהוי דוברים
"""

import runpod
import os
import tempfile
import base64
import io
import torch
import logging
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment

# הגדרת לוגים
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# טעינת token מהסביבה
HF_TOKEN = os.getenv("HF_TOKEN")

# משתנים גלובליים למודלים
asr = None
dia = None


def load_models():
    """טעינת המודלים פעם אחת"""
    global asr, dia

    if asr is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Loading models on: {device}")

        # טעינת Whisper
        model_size = os.getenv("WHISPER_MODEL", "small")
        logger.info(f"Loading Whisper model: {model_size}")
        asr = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )

        # טעינת Pyannote
        logger.info("Loading Pyannote...")
        dia = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=HF_TOKEN
        )
        if device == "cuda":
            dia.to(torch.device("cuda"))

        logger.info("✅ Models loaded successfully!")


def process_audio_to_wav(audio_data: bytes) -> str:
    """המרת אודיו ל-WAV 16kHz מונו"""
    audio = AudioSegment.from_file(io.BytesIO(audio_data))
    audio = audio.set_frame_rate(16000).set_channels(1)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(tmp.name, format="wav")
    return tmp.name


def handler(job):
    """
    RunPod handler function
    
    Input format:
    {
        "input": {
            "audio_base64": "base64_encoded_audio_string",
            "language": "he"
        }
    }
    """
    try:
        job_input = job.get("input", {})
        audio_base64 = job_input.get("audio_base64")

        if not audio_base64:
            return {"error": "Missing audio_base64 in input"}

        language = job_input.get("language", "he")

        load_models()

        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            return {"error": f"Failed to decode base64: {str(e)}"}

        wav_path = None
        try:
            wav_path = process_audio_to_wav(audio_data)

            logger.info("Starting transcription...")
            segments, info = asr.transcribe(
                wav_path,
                beam_size=5,
                language=language,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            transcription = []
            for segment in segments:
                transcription.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip()
                })

            logger.info("Starting speaker diarization...")
            diarization = dia(wav_path)

            speakers = []
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                speakers.append({
                    "start": round(float(turn.start), 2),
                    "end": round(float(turn.end), 2),
                    "speaker": speaker_label
                })

            logger.info("Merging results...")
            final_results = []
            for trans_seg in transcription:
                best_speaker = "SPEAKER_UNKNOWN"
                best_overlap = 0.0

                for spk_seg in speakers:
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

            return {
                "status": "success",
                "language": language,
                "results": final_results,
                "speakers_count": len(set(s["speaker"] for s in final_results))
            }

        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # מאפשר בדיקה מקומית
    print("🔍 Local test mode - no audio provided")
else:
    # נקודת כניסה ל-RunPod
    runpod.serverless.start({"handler": handler})
