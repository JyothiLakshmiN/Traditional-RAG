import streamlit as st
import os
import shutil
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

st.set_page_config(
    page_title="RAG Notebook LLM",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title ---
st.markdown("""
<div style='text-align: center;'>
    <h1 style='color:#6A0DAD;'>📚 RAG Notebook LLM</h1>
    <p style='font-size:18px;color:#555;'>Upload your documents (PDF, TXT, JSON) and ask questions in real time!</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Directories ---
BASE_DATA_DIR = "data"
BASE_FAISS_DIR = "faiss_store"
SESSION_DIR = "current_session"
SESSION_DATA_DIR = os.path.join(BASE_DATA_DIR, SESSION_DIR)
SESSION_FAISS_DIR = os.path.join(BASE_FAISS_DIR, SESSION_DIR)
os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(BASE_FAISS_DIR, exist_ok=True)

# --- Persist state ---
if "store" not in st.session_state:
    st.session_state.store = None
if "rag_search" not in st.session_state:
    st.session_state.rag_search = RAGSearch()

rag_search = st.session_state.rag_search
store = st.session_state.store

# --- Sidebar Upload ---
with st.sidebar:
    st.markdown("## 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or JSON",
        type=["pdf", "txt", "json"],
        accept_multiple_files=True
    )

    st.markdown("---")

# --- Handle uploads ---
if uploaded_files:
    # Clean previous session
    if os.path.exists(SESSION_DATA_DIR):
        shutil.rmtree(SESSION_DATA_DIR)
    if os.path.exists(SESSION_FAISS_DIR):
        shutil.rmtree(SESSION_FAISS_DIR)
    os.makedirs(SESSION_DATA_DIR, exist_ok=True)
    os.makedirs(SESSION_FAISS_DIR, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(SESSION_DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"✅ Uploaded {len(uploaded_files)} file(s) successfully!")

    # Index documents
    with st.spinner("🔧 Indexing your documents..."):
        docs = load_all_documents(SESSION_DATA_DIR)
        store = FaissVectorStore(persist_dir=SESSION_FAISS_DIR)
        store.build_from_documents(docs)
        rag_search.set_vector_store(store)
        st.session_state.store = store

    st.sidebar.success("Documents indexed successfully! You can now query them.")
else:
    st.sidebar.info("📄 Upload files to begin.")

st.divider()

# --- Query Section ---
st.markdown("## 💬 Ask a Question About Your Documents")
query_col, submit_col = st.columns([4,1])
with query_col:
    query = st.text_input("Enter your question here", placeholder="Type your question...")
with submit_col:
    run_button = st.button("🔎 Search & Generate Answer", use_container_width=True)

if run_button:
    if not query.strip():
        st.warning("Please enter a question.")
    elif store is None:
        st.error("⚠️ Please upload and index documents first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            answer = rag_search.search_and_summarize(query, top_k=3)

        st.markdown(f"""
        <div style='background-color:#F5F5F5; padding:20px; border-radius:15px; box-shadow:2px 2px 8px rgba(0,0,0,0.1);'>
            <h4 style='color:#6A0DAD;'>Answer:</h4>
            <div style='font-size:16px; color:#000000;'>{answer}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- Footer ---
st.markdown("""
<div style='text-align:center; margin-top:40px; color:#888; font-size:14px;'>
    ✨ Powered by <strong>Jyothi Lakshmi</strong>
</div>
""", unsafe_allow_html=True)
