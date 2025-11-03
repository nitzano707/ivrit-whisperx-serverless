import whisperx
import os

device = "cuda" if os.environ.get("RUNPOD_GPU_COUNT") else "cpu"

def run_diarization(audio_path: str):
    print("🔹 Loading diarization pipeline...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
    print("🔹 Loading audio...")
    audio = whisperx.load_audio(audio_path)
    print("🔹 Running diarization...")
    diarization = diarize_model(audio)

    results = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": speaker
        })
    return results
