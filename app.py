import os
import subprocess
import asyncio
from pydub import AudioSegment
from faster_whisper import WhisperModel
import whisperx
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------
# הגדרות כלליות
# ----------------------------------------------------------
device = "cuda" if os.environ.get("RUNPOD_GPU_COUNT") else "cpu"

model_transcribe = WhisperModel("ivrit-ai/faster-whisper-v2-d4", device=device)
model_diar = whisperx.DiarizationPipeline(use_auth_token=None, device=device)

# ----------------------------------------------------------
# שלב 1: המרת וידאו לאודיו (אם צריך)
# ----------------------------------------------------------
def extract_audio(input_path):
    audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ], check=True)
    return audio_path

# ----------------------------------------------------------
# שלב 2: פיצול אודיו למקטעים
# ----------------------------------------------------------
def split_audio(path, chunk_length_ms=60000, overlap_ms=500):
    audio = AudioSegment.from_file(path)
    chunks = []
    for start in range(0, len(audio), chunk_length_ms - overlap_ms):
        end = min(len(audio), start + chunk_length_ms)
        chunk = audio[start:end]
        chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{start//1000}_{end//1000}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

# ----------------------------------------------------------
# שלב 3: תמלול עם IVRIT-AI
# ----------------------------------------------------------
def transcribe_chunk(chunk_path):
    segments, _ = model_transcribe.transcribe(chunk_path, beam_size=5)
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]

# ----------------------------------------------------------
# שלב 4: זיהוי דוברים עם WhisperX
# ----------------------------------------------------------
def diarize_chunk(chunk_path):
    audio = whisperx.load_audio(chunk_path)
    diar = model_diar(audio)
    return [
        {"start": seg.start, "end": seg.end, "speaker": label}
        for seg, _, label in diar.itertracks(yield_label=True)
    ]

# ----------------------------------------------------------
# שלב 5: מיזוג תוצאות לפי חפיפת זמן
# ----------------------------------------------------------
def merge(transcription, diarization):
    results = []
    for t in transcription:
        best, overlap = "SPEAKER_UNKNOWN", 0
        for s in diarization:
            start, end = max(t["start"], s["start"]), min(t["end"], s["end"])
            o = max(0, end - start)
            if o > overlap:
                overlap, best = o, s["speaker"]
        results.append({**t, "speaker": best})
    return results

# ----------------------------------------------------------
# שלב 6: תהליך מלא
# ----------------------------------------------------------
def process_audio(input_path):
    if input_path.lower().endswith((".mp4", ".mov", ".mkv")):
        input_path = extract_audio(input_path)

    chunks = split_audio(input_path)
    print(f"🔹 פוצלו {len(chunks)} מקטעים לעיבוד...")

    all_transcripts, all_speakers = [], []
    with ThreadPoolExecutor() as executor:
        trans_results = list(executor.map(transcribe_chunk, chunks))
        diar_results = list(executor.map(diarize_chunk, chunks))

    for t, d in zip(trans_results, diar_results):
        all_transcripts.extend(t)
        all_speakers.extend(d)

    final = merge(all_transcripts, all_speakers)

    # ניקוי קבצים זמניים
    for f in chunks:
        os.remove(f)

    return final
