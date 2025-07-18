# Groq-Powered PDF and YouTube Search Assistant
#
# This script uses the fast Groq cloud API to provide direct answers
# from a PDF, in addition to searching YouTube for relevant videos.

# --- Installation (Required) ---
# pip install openai speechrecognition pyaudio google-api-python-client python-dotenv PyMuPDF groq

# --- API Key Setup ---
# Ensure your .env file in this folder contains your keys:
# OPENAI_API_KEY="your_openai_key_for_whisper"
# YOUTUBE_API_KEY="your_google_cloud_key_here"
# GROQ_API_KEY="your_groq_api_key_here"

import os
import speech_recognition as sr
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import fitz  # PyMuPDF
from groq import Groq

# Load environment variables from the .env file
load_dotenv()


# --- 1. Speech-to-Text (Whisper) ---

def listen_and_transcribe():
    """Captures and transcribes audio using Whisper."""
    try:
        client = OpenAI() # Uses OPENAI_API_KEY from .env
        if not client.api_key: raise ValueError
    except (ValueError, Exception):
        print("ERROR: OpenAI API key not found in your .env file.")
        return None

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("\n🎤 Please speak your query now...")
        try:
            audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Audio captured. Transcribing...")
            temp_audio_path = "temp_query.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data.get_wav_data())
            with open(temp_audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )
            transcribed_text = transcription.text
            print(f"✅ Transcription: '{transcribed_text}'")
            return transcribed_text
        except sr.WaitTimeoutError:
            print("🛑 Listening timed out.")
            return None
        except Exception as e:
            print(f"🛑 An error occurred during transcription: {e}")
            return None
        finally:
            if os.path.exists("temp_query.wav"):
                os.remove("temp_query.wav")


# --- 2. PDF Text Extraction ---

def extract_text_from_pdf(pdf_path: str, query: str):
    """Finds pages in the PDF containing the query and extracts their full text."""
    print(f"\n📄 Searching in PDF: '{os.path.basename(pdf_path)}'...")
    relevant_text = ""
    pages_found = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                # Search for the query on each page
                if page.search_for(query):
                    # If found, add the page number to our list and extract the text
                    if page_num not in pages_found:
                        relevant_text += f"\n--- Content from Page {page_num} ---\n"
                        relevant_text += page.get_text("text")
                        pages_found.append(page_num)
    except FileNotFoundError:
        print(f"🛑 ERROR: The file was not found at '{pdf_path}'")
        return None
    except Exception as e:
        print(f"🛑 An error occurred while reading the PDF: {e}")
        return None
        
    if not relevant_text:
        print(f"No results found for '{query}' in the PDF.")
    else:
        print(f"✅ Found relevant information on pages: {', '.join(map(str, pages_found))}")
    return relevant_text


# --- 3. LLM Answer Generation (Groq) ---

def get_answer_from_groq(context: str, query: str):
    """
    Generates a direct answer from PDF context using the Groq API.
    """
    if not context:
        return "I could not find any relevant information in the PDF to answer your question."

    prompt = f"""
    You are a helpful assistant for an equipment operator. 
    Based ONLY on the following text extracted from a technical manual, 
    provide a clear and direct answer to the operator's question.
    If the context does not contain the answer, state that clearly.
    Do not use any information outside of the provided text.

    CONTEXT FROM MANUAL:
    ---
    {context}
    ---

    OPERATOR'S QUESTION: "{query}"

    ANSWER:
    """

    print("\n⚡ Synthesizing an answer with fast Groq API...")
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("🛑 ERROR: GROQ_API_KEY not found in .env file.")
            return "Groq API key is missing."
        
        client = Groq(api_key=groq_api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", # A fast and capable model on Groq
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"🛑 Groq API error: {e}")
        return "Failed to get response from Groq."


# --- 4. YouTube Video Search ---

def search_youtube(query: str, max_results=3):
    """Searches YouTube for videos based on the text query."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key: return []
    print(f"\n▶️  Searching YouTube for: '{query}'...")
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        search_response = youtube.search().list(q=query, part='snippet', maxResults=max_results, type='video').execute()
        videos = [{'title': item['snippet']['title'], 'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"} for item in search_response.get('items', [])]
        if videos: print(f"✅ Found {len(videos)} videos on YouTube.")
        else: print("No video results found.")
        return videos
    except Exception as e:
        print(f"🛑 An error occurred during YouTube search: {e}")
        return []


# --- Main Application Logic ---

def main():
    """The main function to run the voice search assistant."""
    print("="*50)
    print("   Groq-Powered Assistant: PDF & YouTube Search   ")
    print("="*50)

    pdf_path = input("Please enter the full path to your PDF manual: ").strip()
    if not os.path.exists(pdf_path):
        print("File not found. Please restart.")
        return

    search_query = listen_and_transcribe()
    
    if search_query:
        pdf_context = extract_text_from_pdf(pdf_path, search_query)
        llm_answer = get_answer_from_groq(pdf_context, search_query)
        youtube_results = search_youtube(search_query)

        print("\n\n" + "="*22 + " RESULTS " + "="*22)
        print("\n📄 Answer from PDF Manual (via Groq):\n")
        print(llm_answer)
        print("\n\n▶️ Supplementary YouTube Videos:")
        if youtube_results:
            for video in youtube_results:
                print(f"   - {video['title']}")
                print(f"     Link: {video['url']}")
        else:
            print("   - No videos found.")
        print("\n" + "="*53)


if __name__ == "__main__":
    main()
