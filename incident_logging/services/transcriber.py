# services/transcriber.py
import openai
from config import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(audio_file):
    """
    Sends audio file to OpenAI Whisper API for transcription.
    The audio_file is a Werkzeug FileStorage object from Flask.
    Returns the transcribed text.
    """
    try:
        # THE FIX: Instead of passing the Flask file object directly, we pass a tuple
        # containing the filename and the file's raw byte content, which the
        # OpenAI library correctly understands.
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio_file.filename, audio_file.read()), # This line is changed
            response_format="text"
        )
        return transcript
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Error: Could not transcribe audio. {e}"