import streamlit as st
import os
import shutil
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

st.set_page_config(page_title="RAG Notebook LLM", page_icon="📚", layout="wide")

st.title("📚 Retrieval-Augmented Generation (RAG) Notebook LLM")
st.write("Upload your documents (PDF, TXT, JSON) and ask questions about them in real time!")

# --- Directories ---
BASE_DATA_DIR = "data"
BASE_FAISS_DIR = "faiss_store"
SESSION_DIR = "current_session"
SESSION_DATA_DIR = os.path.join(BASE_DATA_DIR, SESSION_DIR)
SESSION_FAISS_DIR = os.path.join(BASE_FAISS_DIR, SESSION_DIR)
os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(BASE_FAISS_DIR, exist_ok=True)

# --- Persist state between reruns ---
if "store" not in st.session_state:
    st.session_state.store = None
if "rag_search" not in st.session_state:
    st.session_state.rag_search = RAGSearch()

rag_search = st.session_state.rag_search
store = st.session_state.store

# --- File upload ---
st.sidebar.header("📁 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, TXT, or JSON files", 
    type=["pdf", "txt", "json"], 
    accept_multiple_files=True
)

if uploaded_files:
    # Clean old session dirs
    if os.path.exists(SESSION_DATA_DIR):
        shutil.rmtree(SESSION_DATA_DIR)
    if os.path.exists(SESSION_FAISS_DIR):
        shutil.rmtree(SESSION_FAISS_DIR)
    os.makedirs(SESSION_DATA_DIR, exist_ok=True)
    os.makedirs(SESSION_FAISS_DIR, exist_ok=True)

    # Save new files
    for uploaded_file in uploaded_files:
        file_path = os.path.join(SESSION_DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.sidebar.success(f" Uploaded {len(uploaded_files)} file(s) successfully!")

    # Auto-build FAISS index
    with st.spinner("🔧 Indexing your uploaded documents..."):
        docs = load_all_documents(SESSION_DATA_DIR)
        store = FaissVectorStore(persist_dir=SESSION_FAISS_DIR)
        store.build_from_documents(docs)
        rag_search.set_vector_store(store)
        st.session_state.store = store

    st.sidebar.success("Documents indexed successfully! You can now query them.")
else:
    st.sidebar.info("📄 Upload PDF, TXT, or JSON files to begin.")

st.divider()

# --- Query section ---
st.subheader("💬 Ask a Question about Your Uploaded Files")
query = st.text_input("Enter your question here")

if st.button("🔎 Search & Generate Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    elif store is None:
        st.error("⚠️ Please upload and index documents first.")
    else:
        with st.spinner("Retrieving relevant context and generating answer..."):
            answer = rag_search.search_and_summarize(query, top_k=3)
        st.success("Answer:")
        st.write(answer)
