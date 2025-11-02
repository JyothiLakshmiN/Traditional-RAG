# Retrieval Augmented Generation (RAG) Pipeline Implementation

This project implements a **complete Retrieval Augmented Generation (RAG)** pipeline, designed to enhance the capabilities of **Large Language Models (LLMs)** by allowing them to reference **external authoritative knowledge bases**.  

RAG pipelines are at the core of most real-world LLM applications today — from enterprise chatbots to domain-specific assistants — as they effectively mitigate **hallucination** and eliminate the need for **expensive fine-tuning**.

---

## Architectural Overview

The RAG system is structured around two main pipelines that work together:

### **1️Data Injection Pipeline**
This pipeline is responsible for preparing and storing the external knowledge base used during retrieval.

| **Component** | **Functionality** | **Key Steps** |
|----------------|-------------------|----------------|
| **Data Injection & Parsing** | Loads and reads structured/unstructured data (PDF, HTML, Excel, SQL, TXT, etc.) | Converts raw data into **LangChain Document Structure** → `{page_content, metadata}` |
| **Chunking** | Splits large documents into smaller chunks for embedding | Uses **RecursiveCharacterTextSplitter** with custom separators, chunk size, and overlap |
| **Embeddings Generation** | Converts text chunks into dense numerical vectors | Uses **SentenceTransformer (all-MiniLM-L6-v2)** → 384-dimensional embeddings |
| **Vector Store DB** | Stores vectors and metadata persistently | Uses **ChromaDB** or **Faiss** for efficient similarity search and retrieval |

---

### **2 Query Retrieval and Augmented Generation Pipeline**
This pipeline handles user queries, performs retrieval, and generates the augmented final response.

| **Step** | **Functionality** | **Key Mechanism** |
|-----------|-------------------|-------------------|
| **Retrieval (Query Processing)** | Converts user query to vector | Uses same embedding model to ensure consistency |
| **Similarity Search** | Finds top relevant documents | Uses **cosine similarity** to retrieve top-K context vectors |
| **Augmentation** | Combines retrieved context with the LLM prompt | Ensures LLM receives both the query and authoritative context |
| **Generation** | Produces grounded, context-aware answers | Uses **ChatGroq (Gemma 2)** or other LLMs for final output |

---

## Modular Implementation Structure

The project follows a **modular class-based architecture** for clarity and reusability:


---

## Key Technologies Used

| **Component** | **Technology / Library** | **Purpose** |
|----------------|---------------------------|--------------|
| **Orchestration** | `LangChain (Core + Community)` | Workflow and document abstraction |
| **LLM Backend** | `ChatGroq (Gemma 2)` | Fast and cost-efficient text generation |
| **Embeddings** | `SentenceTransformer (all-MiniLM-L6-v2)` | 384-dimensional text embeddings |
| **Vector Store** | `ChromaDB` / `Faiss` | Persistent storage for embeddings |
| **Document Loaders** | `PyPDFLoader`, `PyMuPDFLoader` | PDF parsing and data ingestion |

---

## Enhanced Features

This implementation can be extended with several advanced features:

- **Source Citation** – Returns metadata like filename and page number for transparency  
- **Confidence Scoring** – Uses similarity scores (1 − distance) to show context relevance  
- **Full Context Return** – Optionally exposes retrieved chunks for debugging or analysis  
- **Multi-format Data Support** – Can process PDF, TXT, HTML, and CSV files seamlessly  

---

## Example Workflow

```python
# 1. Process and load PDFs
all_pdf_documents = process_all_pdfs("../data")

# 2. Split into text chunks
chunks = split_documents(all_pdf_documents)

# 3. Generate embeddings
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings(
    [chunk.page_content for chunk in chunks]
)

# 4. Store in ChromaDB
vectorstore = VectorStore()
vectorstore.add_documents(chunks, embeddings)


# Clone the repository
git clone https://github.com/your-username/RAG-Pipeline.git
cd RAG-Pipeline

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
# OR
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
