import os
import time
import pygame

from .config import USER_NAME
from .clients import gcp_tts_client

audio_file_counter = 0

def speak_response(text: str):
    global audio_file_counter
    print(f"மதி: {text}")
    audio_file_counter += 1
    speech_file_path = f"response_{audio_file_counter}.mp3"
    try:
        from google.cloud import texttospeech
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