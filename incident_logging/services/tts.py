# services/tts.py
import requests
from config import ELEVENLABS_API_KEY

# A good, standard voice from ElevenLabs. You can find more voice IDs on their website.
VOICE_ID = "21m00Tcm4TlvDq8ikWAM" 
ELEVENLABS_API_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

def text_to_speech_elevenlabs(text: str):
    """
    Converts text to speech using ElevenLabs API and returns the audio content.
    """
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        response = requests.post(ELEVENLABS_API_URL, json=data, headers=headers)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error calling ElevenLabs API: {e}")
        return None