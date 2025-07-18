# python learning assisstant/assistant_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# --- LangChain components ---
# Ensure you have these installed in your venv:
# pip install langchain langchain-groq faiss-cpu sentence-transformers
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
# We load the DB and create the RAG chain once when the server starts
DB_FAISS_PATH = 'faiss_index'
RAG_CHAIN = None

def load_and_setup():
    """Loads the vector DB and prepares the RAG chain."""
    global RAG_CHAIN
    
    # 1. Load Vector DB
    print("Loading knowledge base...")
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

    # 2. Setup RAG Chain
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

# --- API Endpoint ---
@app.route('/learning/ask', methods=['POST'])
def ask_assistant():
    """Accepts a question in JSON and returns a text answer."""
    if not RAG_CHAIN:
        return jsonify({"error": "Assistant is not ready. Check server logs."}), 503

    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "A 'question' is required."}), 400
        
    try:
        print(f"🧠 Processing question: '{question}'")
        # Invoke the RAG chain to get the answer from the knowledge base
        answer = RAG_CHAIN.invoke(question)
        print(f"✅ Generated answer.")
        
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        print(f"🛑 Error during RAG chain invocation: {e}")
        return jsonify({"error": "Failed to generate an answer."}), 500

if __name__ == '__main__':
    # Load the models before starting the server
    if load_and_setup():
        # Run on port 5004
        app.run(host='0.0.0.0', port=5004, debug=True)
    else:
        print("\nServer startup failed due to errors. Please check the logs.")