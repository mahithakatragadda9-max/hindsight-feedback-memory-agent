import io
from groq import Groq
from config import GROQ_API_KEY


class VoiceUnavailable(Exception):
    pass


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    if not GROQ_API_KEY:
        raise VoiceUnavailable("GROQ_API_KEY is not configured; cannot transcribe audio.")
    if not audio_bytes:
        raise VoiceUnavailable("No audio received.")

    client = Groq(api_key=GROQ_API_KEY)
    buf = io.BytesIO(audio_bytes)
    buf.name = filename

    try:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=buf,
        )
    except Exception as e:
        raise VoiceUnavailable(f"Transcription failed: {e}") from e

    text = getattr(resp, "text", "") or ""
    if not text.strip():
        raise VoiceUnavailable("Transcription returned empty text.")
    return text.strip()
