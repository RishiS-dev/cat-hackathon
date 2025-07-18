import subprocess
import threading

# Replace with your user's name if needed
USER_NAME = "manikandan"

# Global variable to track the music process
current_music_process = None

def speak_response(text: str):
    """Dummy function: replace with TTS or print as needed."""
    print(f"Assistant: {text}")

def stop_all_music():
    """Stop any active music streaming process."""
    global current_music_process
    if current_music_process:
        try:
            print("Stopping active ffplay stream...")
            current_music_process.terminate()
            current_music_process.wait(timeout=2)
        except Exception as e:
            print(f"Forcing kill on ffplay stream: {e}")
            current_music_process.kill()
        finally:
            current_music_process = None

def stream_with_ffplay(search_query: str):
    """
    Streams audio from YouTube using yt-dlp and ffplay.
    """
    global current_music_process
    stop_all_music()

    try:
        speak_response(f"{USER_NAME}, streaming '{search_query}'...")

        # Get the direct audio stream URL from YouTube
        url_command = [
            'yt-dlp', '--get-url', '--format', 'bestaudio[ext=m4a]/bestaudio',
            '--no-playlist', f'ytsearch1:{search_query}'
        ]
        result = subprocess.run(url_command, capture_output=True, text=True, timeout=20)

        if result.returncode == 0 and result.stdout.strip():
            stream_url = result.stdout.strip().split('\n')[0]
            ffplay_command = [
                'ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', stream_url
            ]
            current_music_process = subprocess.Popen(
                ffplay_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            speak_response(f"{USER_NAME}, music ஆரம்பிச்சுடுத்து!")

            # Reset process when done
            def monitor_process(proc):
                global current_music_process
                proc.wait()
                if current_music_process == proc:
                    current_music_process = None
                    print("Streaming process finished.")

            threading.Thread(target=monitor_process, args=(current_music_process,), daemon=True).start()
            return True
        else:
            print(f"Failed to get stream URL. stderr: {result.stderr}")
            speak_response("Song stream URL not found.")
            return False

    except subprocess.TimeoutExpired:
        speak_response(f"{USER_NAME}, streaming timeout ஆயிடுத்து. வேற பாட்டு try பண்ணுங்க.")
        return False
    except FileNotFoundError:
        speak_response("FFmpeg/ffplay or yt-dlp not found. Please install them.")
        return False
    except Exception as e:
        print(f"FFplay streaming error: {e}")
        speak_response("Streaming error.")
        return False

# --- Test usage ---
if __name__ == "__main__":
    search_query = input("Enter song or artist to stream: ")
    stream_with_ffplay(search_query)