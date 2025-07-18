# Companion/companion_server.py
import os
import io
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import openai
import groq
from google.cloud import texttospeech
import base64 # Needed to send audio back to the frontend

# --- 1. SETUP AND INITIALIZATION ---
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(__file__), "gcp_credentials.json")

# --- Initialize Clients ---
try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
    gcp_tts_client = texttospeech.TextToSpeechClient()
except Exception as e:
    print(f"🔴 FATAL: Could not initialize API clients. Check your .env and credentials. Error: {e}")
    exit()

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- 2. CORE LOGIC (Refactored for API) ---

def transcribe_audio_from_file(audio_file):
    """Transcribes an audio file object using Whisper."""
    try:
        # The 'audio_file' comes directly from the HTTP request
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, language="ta"
        )
        print(f"✅ Transcription: {transcript.text}")
        return transcript.text
    except Exception as e:
        print(f"🔴 Whisper transcription error: {e}")
        return None

def get_llm_response(text: str, user_name="Operator"):
    """Gets a conversational response from Groq."""
    system_prompt = (
        f"நீங்கள் 'மதி', {user_name} என்ற நபரின் உதவியாளர். "
        "எளிமையான, நட்பான தமிழில் பேசுங்கள். குறுகிய பதில்கள் கொடுங்கள்."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages, model="llama3-8b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"🔴 Groq API error: {e}")
        return "மன்னிக்கவும், ஒரு சிறிய சிக்கல் உள்ளது."

def synthesize_speech(text: str):
    """Converts text to speech audio bytes (MP3)."""
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ta-IN", name="ta-IN-Wavenet-B",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = gcp_tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        # Return the raw audio content
        return response.audio_content
    except Exception as e:
        print(f"🔴 Google TTS Error: {e}")
        return None

def get_song_title_from_llm(text: str):
    """Extracts a Youtube query from a music request."""
    # This function is good as-is, we just call it from the endpoint.
    system_prompt = (
        "You are an expert music search query optimizer... [Your full prompt here]"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages, model="llama3-8b-8192", temperature=0.1
        )
        search_query = chat_completion.choices[0].message.content.strip().replace('"', '').replace("'", "")
        print(f"✅ LLM optimized music query: '{search_query}'")
        return search_query
    except Exception as e:
        print(f"🔴 LLM song title extraction failed: {e}")
        return text # Fallback to the original text

# --- 3. FLASK API ENDPOINT ---

@app.route('/companion/ask', methods=['POST'])
def handle_ask():
    # The frontend will send the audio as a file in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files['audio']
    
    # 1. Transcribe the user's audio
    user_input_text = transcribe_audio_from_file(audio_file)
    if not user_input_text:
        return jsonify({"error": "Could not understand audio"}), 500

    response_text = ""
    response_audio_b64 = ""
    action = "conversation"
    action_payload = {}

    # 2. Decide the action: Music or Conversation
    music_keywords = ["இசை", "பாடல்", "பாட்டு", "music", "play", "கேட்க", "போடு"]
    if any(keyword in user_input_text.lower() for keyword in music_keywords):
        action = "play_music"
        search_query = get_song_title_from_llm(user_input_text)
        response_text = f"சரி, '{search_query}' பாடலைத் தேடுகிறேன்."
        action_payload = {"youtube_query": search_query}
    else:
        # It's a regular conversation
        response_text = get_llm_response(user_input_text)

    # 3. Synthesize the response text to audio
    audio_bytes = synthesize_speech(response_text)
    if audio_bytes:
        # Encode the audio bytes into Base64 to send it in JSON
        response_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

    # 4. Send everything back to the frontend
    return jsonify({
        "user_transcription": user_input_text,
        "ai_response_text": response_text,
        "ai_response_audio_b64": response_audio_b64, # The frontend will decode and play this
        "action": action,
        "action_payload": action_payload
    })


if __name__ == '__main__':
    print("🚀 Starting Companion Server...")
    # Run on a new port, e.g., 5005
    app.run(host='0.0.0.0', port=5005, debug=True)