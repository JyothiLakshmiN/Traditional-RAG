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

def summarize_documents(docs, rag_search):
    combined_text = "\n\n".join([d.page_content for d in docs])[:6000]  # limit context
    prompt = f"Summarize the following documents in a clear, concise way:\n\n{combined_text}\n\nSummary:"
    return rag_search.llm.invoke(prompt).content

# --- Title ---
st.markdown("""
<div style='text-align: center;'>
    <h1 style='color:#6A0DAD;'>📚 RAG Notebook LLM (Conversational)</h1>
    <p style='font-size:18px;color:#555;'>Upload documents and chat with them!</p>
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


# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # conversational history
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

# --- File Upload & Indexing ---
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

    # Indexing
    with st.spinner("Indexing your documents..."):
        docs = load_all_documents(SESSION_DATA_DIR)
        store = FaissVectorStore(persist_dir=SESSION_FAISS_DIR)
        store.build_from_documents(docs)
        rag_search.set_vector_store(store)
        st.session_state.store = store

    st.sidebar.success("Documents indexed successfully!")
    # Clear previous chat history when new docs are uploaded
    st.session_state.messages = []
    with st.spinner("📝 Generating summary..."):
        summary = summarize_documents(docs, rag_search)
        st.session_state["summary"] = summary

    st.success("Summary Ready!")
    st.markdown(f"""
    <div style='padding:15px; border-radius:10px;'>
    <h4 style='color:#6A0DAD;'>📄 Document Summary:</h4>
    <p style='font-size:16px; color:#000000 !important;'>{summary}</p>
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.sidebar.info("📄 Upload files to begin.")

# --- Display chat history ---
for msg in st.session_state.messages:
    role, content = msg
    with st.chat_message(role):
        st.write(content)

# --- Chat input box ---
user_input = st.chat_input("Ask something about your documents...")

if user_input:
    docs_exist = os.path.exists(SESSION_DATA_DIR) and len(os.listdir(SESSION_DATA_DIR)) > 0
    if not docs_exist or store is None:
        st.error("No documents found. Please upload and index documents first.")
        # Clear chat history and summary
        st.session_state.messages = []
        st.session_state["summary"] = ""
        st.session_state.store = None
    else:
        # Display user message
        st.session_state.messages.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_search.search_and_summarize(user_input, top_k=3)
                st.write(answer)

        # Save to conversation memory
        st.session_state.messages.append(("assistant", answer))

st.divider()

# --- Footer ---
st.markdown("""
<div style='text-align:center; margin-top:40px; color:#888; font-size:14px;'>
    Powered by <strong>Jyothi Lakshmi</strong>
</div>
""", unsafe_allow_html=True)
