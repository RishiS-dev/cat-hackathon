from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import base64 # Needed to send audio back to the frontend

# --- New Imports for Audio, TTS, and YouTube ---
from openai import OpenAI
from googleapiclient.discovery import build
from gtts import gTTS

# --- LangChain components ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Global Variables ---
# Get the absolute path of the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Join that path with the relative 'faiss_index' folder name
DB_FAISS_PATH = os.path.join(SCRIPT_DIR, 'faiss_index')
RAG_CHAIN = None
OPENAI_CLIENT = None

# --- Helper Function for YouTube Search (from your other script) ---
def search_youtube(query: str, max_results=3):
    """Searches YouTube for videos based on the text query."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Warning: YOUTUBE_API_KEY not found. Skipping YouTube search.")
        return []
        
    print(f"▶️  Searching YouTube for: '{query}'...")
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

def load_and_setup():
    """Loads the vector DB, prepares the RAG chain, and sets up API clients."""
    global RAG_CHAIN, OPENAI_CLIENT
    
    # 1. Setup OpenAI Client
    try:
        OPENAI_CLIENT = OpenAI()
        if not OPENAI_CLIENT.api_key: raise ValueError("OpenAI API Key not found")
        print("✅ OpenAI client is ready.")
    except Exception as e:
        print(f"🔴 FATAL: Could not initialize OpenAI client. Is OPENAI_API_KEY set? Error: {e}")
        return False

    # 2. Load Vector DB
    print(f"Loading knowledge base from: {DB_FAISS_PATH}")
    if not os.path.exists(DB_FAISS_PATH):
        print(f"🔴 FATAL: Vector database folder '{DB_FAISS_PATH}' not found.")
        return False
        
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ Knowledge base loaded.")
    except Exception as e:
        print(f"🔴 FATAL: Failed to load vector DB. Error: {e}")
        return False

    # 3. Setup RAG Chain
    print("Setting up RAG chain...")
    try:
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

        RAG_CHAIN = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("✅ RAG chain is ready.")
        return True
    except Exception as e:
        print(f"🔴 FATAL: Could not create RAG chain. Is GROQ_API_KEY set? Error: {e}")
        return False

# --- API Endpoints ---

@app.route('/learning/ask_text', methods=['POST'])
def ask_assistant_text():
    """Accepts a text question in JSON and returns a text answer."""
    if not RAG_CHAIN:
        return jsonify({"error": "Assistant is not ready. Check server logs."}), 503

    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "A 'question' is required."}), 400
        
    try:
        print(f"🧠 Processing TEXT question: '{question}'")
        answer = RAG_CHAIN.invoke(question)
        print(f"✅ Generated answer.")
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        print(f"🛑 Error during RAG chain invocation: {e}")
        return jsonify({"error": "Failed to generate an answer."}), 500

@app.route('/learning/ask_audio', methods=['POST'])
def ask_assistant_audio():
    """Accepts an audio file, transcribes it, gets an answer, and returns it with TTS audio."""
    if not RAG_CHAIN or not OPENAI_CLIENT:
        return jsonify({"error": "Assistant is not ready. Check server logs."}), 503

    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file found in the request."}), 400

    audio_file = request.files['audio_data']
    temp_audio_path = "temp_query_from_web.webm"
    audio_file.save(temp_audio_path)

    try:
        # 1. Transcribe Audio to Text
        print("🎤 Transcribing audio...")
        with open(temp_audio_path, "rb") as audio:
            transcription = OPENAI_CLIENT.audio.transcriptions.create(
                model="whisper-1", file=audio
            )
        question = transcription.text
        print(f"✅ You said: '{question}'")

        # 2. Get Answer from RAG Chain
        print(f"🧠 Processing AUDIO question: '{question}'")
        answer = RAG_CHAIN.invoke(question)
        print(f"✅ Generated answer.")

        # 3. Search for YouTube videos
        youtube_results = search_youtube(question)

        # 4. Convert Answer to Speech (TTS)
        print("▶️ Generating audio response...")
        tts = gTTS(text=answer, lang='en')
        tts_file_path = "response.mp3"
        tts.save(tts_file_path)

        # 5. Read audio file and encode it in Base64
        with open(tts_file_path, "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # 6. Clean up temporary files
        os.remove(tts_file_path)
        
        # 7. Send everything back to the frontend
        return jsonify({
            "question": question, 
            "answer": answer,
            "audio_response": audio_base64,
            "youtube_links": youtube_results
        })

    except Exception as e:
        print(f"🛑 Error during audio processing: {e}")
        return jsonify({"error": "Failed to process audio and generate an answer."}), 500
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


if __name__ == '__main__':
    if load_and_setup():
        # Renamed old endpoint to /ask_text to avoid conflicts
        # New endpoint is /ask_audio
        app.run(host='0.0.0.0', port=5004, debug=True)
    else:
        print("\nServer startup failed due to errors. Please check the logs.")
