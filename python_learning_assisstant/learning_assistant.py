# RAG-Powered Conversational Assistant from a Saved Vector Database
#
# This script loads a pre-built FAISS vector database from a local folder,
# listens for questions, uses Groq to generate an answer, and also searches
# for supplementary YouTube videos.

# --- Installation (Required) ---
# pip install openai speechrecognition pyaudio google-api-python-client python-dotenv PyMuPDF groq langchain langchain-groq faiss-cpu sentence-transformers gtts playsound

# --- Setup (CRITICAL) ---
# 1. First, run your separate script to create the 'faiss_index' folder from your knowledge base.
# 2. Your `.env` file must be present and contain your API keys:
#    OPENAI_API_KEY="your_openai_key_for_whisper"
#    YOUTUBE_API_KEY="your_google_cloud_key_here"
#    GROQ_API_KEY="your_groq_api_key_here"

import os
import speech_recognition as sr
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
from playsound import playsound

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Load API Keys ---
load_dotenv()

# Define the path to the saved vector database
DB_FAISS_PATH = 'faiss_index'

def speak(text):
    """
    Converts text to speech and plays it.
    """
    try:
        tts = gTTS(text=text, lang='en')
        tts_file = "response.mp3"
        tts.save(tts_file)
        print("▶️ Speaking...")
        playsound(tts_file)
        os.remove(tts_file) # Clean up the audio file
    except Exception as e:
        print(f"🛑 Could not play the audio. Error: {e}")

def load_vector_db():
    """
    Loads the vector database from the local folder.
    """
    if not os.path.exists(DB_FAISS_PATH):
        print(f"ERROR: The vector database folder '{DB_FAISS_PATH}' was not found.")
        print("Please run the database creation script first.")
        return None

    print(f"Loading knowledge base from '{DB_FAISS_PATH}'...")
    # The embeddings model must be the same as the one used for creation
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Knowledge base loaded successfully.")
        return db
    except Exception as e:
        print(f"🛑 Failed to load the vector database. Error: {e}")
        return None


def listen_and_transcribe():
    """Captures and transcribes audio using Whisper."""
    try:
        client = OpenAI()
        if not client.api_key: raise ValueError
    except (ValueError, Exception):
        print("ERROR: OpenAI API key not found in your .env file.")
        return None

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("\n🎤 How can I help you? (Say 'goodbye' to exit)")
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
            print(f"✅ You said: '{transcribed_text}'")
            return transcribed_text
        except sr.WaitTimeoutError:
            print("🛑 Listening timed out. Please try again.")
            return None
        except Exception as e:
            print(f"🛑 An error occurred during transcription: {e}")
            return None
        finally:
            if os.path.exists("temp_query.wav"):
                os.remove("temp_query.wav")

def setup_rag_chain(db):
    """
    Sets up the LangChain RAG pipeline using the vector database.
    """
    if not db:
        return None
        
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    retriever = db.as_retriever()
    
    template = """
    You are an assistant for a Caterpillar excavator operator.
    Answer the question based ONLY on the following context from the provided manual.
    If you don't know the answer from the context, state that clearly.
    Keep the answer concise and use step-by-step instructions if possible.

    CONTEXT: {context}

    QUESTION: {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # This chain ties together the retriever, prompt, and LLM
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def search_youtube(query: str, max_results=3):
    """Searches YouTube for videos based on the text query."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Warning: YOUTUBE_API_KEY not found in .env file. Skipping YouTube search.")
        return []
        
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

def main():
    """The main function to run the RAG assistant."""
    print("="*55)
    print("   Conversational RAG Assistant (using Groq)   ")
    print("="*55)
    
    # Step 1: Load the pre-built Vector DB
    vector_db = load_vector_db()
    if not vector_db:
        return

    # Step 2: Set up the RAG chain
    rag_chain = setup_rag_chain(vector_db)
    if not rag_chain:
        return

    speak("Hello! The knowledge base is loaded. How can I help you today?")

    while True:
        question = listen_and_transcribe()
        
        if question:
            if "goodbye" in question.lower() or "stop" in question.lower():
                print("Assistant shutting down. Stay safe!")
                speak("Goodbye!")
                break

            # Invoke the RAG chain to get an answer from the PDF
            print("\n🧠 Checking the knowledge base for you...")
            llm_answer = rag_chain.invoke(question)
            
            # Also search for supplementary YouTube videos
            youtube_results = search_youtube(question)

            print("\n\n" + "="*22 + " RESULTS " + "="*22)
            print("\n📄 Answer from Knowledge Base (via Groq):\n")
            print(llm_answer)
            
            # Speak the primary answer from the PDF
            speak(llm_answer)
            
            print("\n\n▶️ Supplementary YouTube Videos:")
            if youtube_results:
                for video in youtube_results:
                    print(f"   - {video['title']}")
                    print(f"     Link: {video['url']}")
            else:
                print("   - No videos found.")
            print("\n" + "="*53)
        else:
            continue

if __name__ == "__main__":
    main()
