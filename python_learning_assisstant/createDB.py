# Create Vector Database Script
#
# This script reads a markdown file, processes the text, creates vector embeddings,
# and saves the resulting FAISS vector database to a local folder.
# Run this script once, or whenever your knowledgebase.md file is updated.

# --- Installation (Required) ---
# pip install python-dotenv langchain faiss-cpu sentence-transformers

# --- Setup (CRITICAL) ---
# 1. Create a file named `knowledgebase.md` in the same folder.
#    Paste all the excavator knowledge base content into this file.

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the name for the local folder where the DB will be saved
DB_FAISS_PATH = 'faiss_index'

def load_knowledge_from_file(file_path="knowledgebase.md"):
    """
    Reads the content from the specified markdown file.
    """
    print(f"Loading knowledge from '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("Knowledge base loaded successfully.")
        return content
    except FileNotFoundError:
        print(f"ERROR: The file '{file_path}' was not found.")
        print("Please create this file in the same directory as the script.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def create_and_save_vector_db(knowledge_content):
    """
    Creates and saves an in-memory vector database from the loaded content.
    """
    if not knowledge_content:
        print("Knowledge content is empty. Aborting database creation.")
        return
        
    print("Creating vector database...")
    
    # Use a text splitter to break the document into smaller, manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(knowledge_content)
    
    # Use a sentence-transformer model for creating the embeddings (vectors)
    # This model runs locally on your CPU and is downloaded automatically.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the FAISS vector store from the text chunks and embeddings
    db = FAISS.from_texts(texts, embeddings)
    
    # Save the vector database to the local folder
    db.save_local(DB_FAISS_PATH)
    print(f"Vector database created and saved successfully to '{DB_FAISS_PATH}/'")

if __name__ == "__main__":
    knowledge = load_knowledge_from_file()
    if knowledge:
        create_and_save_vector_db(knowledge)

