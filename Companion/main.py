import os
import io
import re
import time
import tempfile
from dotenv import load_dotenv

import openai
import groq
import sounddevice as sd
from scipy.io.wavfile import write
from google.cloud import texttospeech
import pygame

from stream import stream_with_ffplay, stop_all_music  # <-- Use the streaming methods from stream.py

# --- 1. SETUP AND INITIALIZATION ---

USER_NAME = "manikandan"
CURRENT_USER_LOGIN = "rishi6-dev"

# Load environment variables
load_dotenv()

# Configure Google Cloud authentication
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"
except FileNotFoundError:
    print("Error: 'gcp_credentials.json' not found.")
    exit()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
groq_client = groq.Groq(api_key=GROQ_API_KEY)
gcp_tts_client = texttospeech.TextToSpeechClient()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

audio_file_counter = 0

# --- 2. MUSIC LOGIC USING STREAM.PY ---

def play_youtube_music(search_query: str):
    """Delegates streaming to stream.py method. Always plays fallback if needed."""
    return stream_with_ffplay(search_query)

# --- 3. CORE FUNCTIONS ---

def record_and_transcribe_tamil_audio(duration=5, fs=44100):
    print(f"\n{USER_NAME}, பேசுங்க... (Listening...)")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("யோசிக்கிறேன்... (Processing...)")
    virtual_file = io.BytesIO()
    write(virtual_file, fs, recording)
    virtual_file.seek(0)
    virtual_file.name = "operator_audio.wav"
    try:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1", file=virtual_file, language="ta"
        )
        print(f"{USER_NAME} சொன்னது: {transcript.text}")
        return transcript.text
    except Exception as e:
        print(f"Whisper பிழை: {e}")
        return None

def get_companion_response(text: str, conversation_history: list):
    if not text:
        return f"{USER_NAME}, நீங்க சொன்னது கேக்கல. மறுபடியும் சொல்லுங்க."
    system_prompt = (
        f"நீங்கள் 'மதி', {USER_NAME} என்ற நபரின் உதவியாளர். "
        "எளிமையான, நட்பான தமிழில் பேசுங்கள். குறுகிய பதில்கள் கொடுங்கள்."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": text}
    ]
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages, model="llama3-8b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {e}")
        return f"மன்னிக்கவும் {USER_NAME}, connection problem."

def speak_response(text: str):
    global audio_file_counter
    print(f"மதி: {text}")
    audio_file_counter += 1
    speech_file_path = f"response_{audio_file_counter}.mp3"
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ta-IN", name="ta-IN-Wavenet-B",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.9, pitch=-3.0, volume_gain_db=3.0
        )
        response = gcp_tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(speech_file_path, "wb") as out:
            out.write(response.audio_content)
        music_was_playing = pygame.mixer.music.get_busy()
        if music_was_playing:
            pygame.mixer.music.pause()
        speech_sound = pygame.mixer.Sound(speech_file_path)
        speech_sound.play()
        while pygame.mixer.get_busy():
            pygame.time.Clock().tick(10)
        if music_was_playing:
            pygame.mixer.music.unpause()
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        if os.path.exists(speech_file_path):
            time.sleep(0.2)
            try:
                os.remove(speech_file_path)
            except Exception as e:
                print(f"Could not remove speech file {speech_file_path}: {e}")

def simple_extract_music_keywords(text: str):
    music_words = ["இசை", "பாடல்", "பாட்டு", "ஒலிபரப்பு", "music", "play", "நிகழ்த்து", "கேட்க", "போடு"]
    cleaned_text = text
    for word in music_words:
        cleaned_text = re.sub(rf'\b{word}\b', '', cleaned_text, flags=re.IGNORECASE)
    return cleaned_text.strip()

def get_song_title_from_llm(text: str):
    print("Using LLM to extract song title...")
    try:
        system_prompt = (
            "You are an expert music search query optimizer. The user will provide a sentence in Tamil or English. "
            "Your only task is to extract the song name, artist, or movie. "
            "Return ONLY the most effective search query for YouTube. "
            "For example, if the user says 'ரஞ்சிதமே பாட்டு போடு அது சின்ன பாட்டு தான்', you must return 'Ranjithame song'. "
            "If they say 'play the latest Anirudh song', return 'Latest Anirudh song'. "
            "Do not add any explanations or conversational text. Just the search query."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.1
        )
        search_query = chat_completion.choices[0].message.content.strip().replace('"', '').replace("'", "")
        print(f"LLM optimized search query: '{search_query}'")
        return search_query
    except Exception as e:
        print(f"LLM song title extraction failed: {e}. Falling back to simple extractor.")
        return simple_extract_music_keywords(text)

def log_session_activity(activity: str):
    current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("session_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"[{current_timestamp}] {USER_NAME} ({CURRENT_USER_LOGIN}) - {activity}\n")

def cleanup_on_exit():
    print("\nCleaning up resources...")
    stop_all_music()
    pygame.mixer.quit()
    import glob
    for pattern in ["response_*.mp3"]:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Removed temp file: {file}")
            except OSError as e:
                print(f"Error removing file {file}: {e}")

def main_loop():
    conversation_history = []
    log_session_activity("Session started.")
    intro_message = f"Hello {USER_NAME}! நான் மதி. Music கேட்கணுமா அல்லது வேற help வேணுமா சொல்லுங்க."
    speak_response(intro_message)
    while True:
        try:
            input(f"மதி கிட்ட பேச Enter அழுத்துங்க | Exit க்கு Ctrl+C...")
            user_input_text = record_and_transcribe_tamil_audio()
            if not user_input_text:
                speak_response(f"{USER_NAME}, clear-ஆ கேக்கல. மறுபடி try பண்ணுங்க.")
                continue
            log_session_activity(f"User input: {user_input_text}")
            goodbye_phrases = ["விடை", "போகிறேன்", "வெளியேறு", "பை", "goodbye", "bye", "முடிக்க"]
            if any(phrase in user_input_text.lower() for phrase in goodbye_phrases):
                speak_response(f"Bye {USER_NAME}! Safe-ஆ இருங்க.")
                log_session_activity("Session ended by user.")
                break
            stop_keywords = ["நிறுத்து", "stop", "pause", "வேண்டாம்"]
            if any(keyword in user_input_text.lower() for keyword in stop_keywords):
                if stop_all_music():
                    speak_response(f"{USER_NAME}, music நிறுத்திட்டேன்.")
                else:
                    speak_response(f"{USER_NAME}, music எதுவும் ஓடலையே.")
                continue
            music_keywords = ["இசை", "பாடல்", "பாட்டு", "ஒலிபரப்பு", "music", "play", "நிகழ்த்து", "கேட்க", "போடு"]
            if any(keyword in user_input_text.lower() for keyword in music_keywords):
                search_query = get_song_title_from_llm(user_input_text)
                if not search_query or len(search_query) < 2:
                    search_query = "fallback"
                log_session_activity(f"Music request (optimized): {search_query}")
                play_youtube_music(search_query)
            else:
                conversation_history.append({"role": "user", "content": user_input_text})
                response_text = get_companion_response(user_input_text, conversation_history)
                conversation_history.append({"role": "assistant", "content": response_text})
                log_session_activity(f"AI response: {response_text}")
                speak_response(response_text)
                if len(conversation_history) > 6:
                    conversation_history = conversation_history[-6:]
        except KeyboardInterrupt:
            log_session_activity("Session interrupted by Ctrl+C.")
            break

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_on_exit()
        print(f"\nBye {USER_NAME}! (மதி shutting down)")