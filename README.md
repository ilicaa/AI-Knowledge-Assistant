# 🤖 AI Knowledge Assistant

A **RAG-based (Retrieval-Augmented Generation)** AI application that answers questions using uploaded documents. Built with Python, Streamlit, FAISS, and LLM integration.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Setup Instructions](#-setup-instructions)
- [How to Run](#-how-to-run)
- [How It Works](#-how-it-works)
- [Performance Strategy](#-performance-strategy)
- [Project Structure](#-project-structure)
- [Limitations & Future Improvements](#-limitations--future-improvements)

---

## ✨ Features

### Core Functionality
- **📄 Document Upload**: Support for PDF, TXT, and Markdown files
- **🔍 Smart Search**: Semantic similarity search using embeddings
- **💬 AI-Powered Answers**: Grounded responses using retrieved context
- **📚 Source Citations**: Clear references to source documents

### User Experience
- **🖥️ Clean Interface**: Modern Streamlit UI
- **⚡ Fast Responses**: Optimized for < 5 second inference
- **🗑️ Document Management**: Add, view, and delete documents
- **📊 Performance Metrics**: Real-time timing information

### Engineering Quality
- **🏗️ Modular Design**: Clean separation of concerns
- **🔄 Error Handling**: Graceful failure handling
- **📝 Logging**: Comprehensive debug logging
- **🧪 Testable**: Unit tests included

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Interface                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Doc Upload   │  │ Question Box │  │ Answer + Sources     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Retrieval    │──│ Context      │──│ LLM Generation       │  │
│  │ (Top-K)      │  │ Building     │  │ (Grounded Answer)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Document         │ │ Vector Store     │ │ LLM Client       │
│ Processor        │ │ (FAISS)          │ │ (OpenAI/Groq)    │
│ - PDF/TXT/MD     │ │ - Embeddings     │ │ - API Calls      │
│ - Chunking       │ │ - Similarity     │ │ - Retry Logic    │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AI-Knowledge-Assistant.git
cd AI-Knowledge-Assistant
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key(s)
# You need at least ONE of:
# - OPENAI_API_KEY (from https://platform.openai.com/api-keys)
# - GROQ_API_KEY (from https://console.groq.com/keys) - Recommended for speed
```

**Example `.env` file:**
```env
# Recommended: Groq (faster, free tier available)
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_PROVIDER=groq

# Or use OpenAI
# OPENAI_API_KEY=sk-your_openai_api_key_here
# LLM_PROVIDER=openai
```

---

## ▶️ How to Run

> ⚠️ **IMPORTANT: Always run commands inside the virtual environment!**
> 
> Before running any command, make sure the venv is activated:
> ```bash
> source venv/bin/activate  # Linux/Mac
> venv\Scripts\activate     # Windows
> ```
> You should see `(venv)` at the beginning of your terminal prompt.

### Option 1: Using the runner script
```bash
python run.py
```

### Option 2: Using Streamlit directly
```bash
streamlit run app/main.py
```

### Running Tests
```bash
# Make sure venv is activated, then:
python -m pytest tests/ -v
```

### Access the Application
Open your browser and navigate to: **http://localhost:8501**

---

## 🔧 How It Works

### 1. Document Ingestion
When you upload a document:

1. **File Parsing**: The document is parsed based on its type (PDF, TXT, or MD)
2. **Text Chunking**: Content is split into overlapping chunks (default: 500 chars with 50 char overlap)
3. **Embedding Generation**: Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`
4. **Storage**: Embeddings and metadata are stored in FAISS (persistent)

```python
# Chunking Strategy
- Chunk size: 500 characters
- Overlap: 50 characters (maintains context between chunks)
- Smart breaks: Tries to split at sentence boundaries
```

### 2. Retrieval Process
When you ask a question:

1. **Query Embedding**: Your question is converted to a vector
2. **Similarity Search**: FAISS finds the top-K most similar chunks (default: 5)
3. **Ranking**: Results are ranked by cosine similarity score

```python
# Retrieval Configuration
- Top-K results: 5 (configurable)
- Similarity metric: Cosine distance
- Max context length: 3000 characters
```

### 3. Answer Generation
The LLM generates an answer using:

1. **Context Building**: Retrieved chunks are formatted with source information
2. **Prompt Engineering**: A carefully designed system prompt ensures grounded answers
3. **Generation**: The LLM produces an answer citing sources
4. **Validation**: If no relevant information is found, it says "I don't know"

### Prompt Approach
```python
System Prompt (Summary):
- Answer ONLY from provided context
- If information not present, say "I don't have enough information"
- Cite sources when referencing information
- Be concise and direct
- Never make up information
```

---

## ⚡ Performance Strategy

### How We Achieve < 5 Second Inference

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| **Fast Embedding Model** | ~0.05s query embedding | `all-MiniLM-L6-v2` (384 dims, 80MB) |
| **Limited Retrieval** | ~0.04-0.07s search time | Top-5 chunks only |
| **Controlled Context** | Faster LLM processing | Max 3000 chars context |
| **Fast LLM (Groq)** | ~0.4-0.8s generation | GPT-OSS-20B on Groq |
| **Persistent Storage** | No re-indexing | FAISS persistence |
| **Singleton Patterns** | No model reloading | Cached model instances |

### Typical Response Time Breakdown
```
┌────────────────────────────────────────┐
│ Query Embedding:     ~0.04s            │
│ Vector Search:       ~0.04s            │
│ Context Building:    ~0.01s            │
│ LLM Generation:      ~0.4-0.6s (Groq)  │
│ ─────────────────────────────          │
│ Total:               ~0.4-0.7s ✓       │
└────────────────────────────────────────┘
```

## 📁 Project Structure

```
AI-Knowledge-Assistant/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Streamlit application entry point
│   ├── config.py             # Centralized configuration
│   ├── document_processor.py # Document parsing and chunking
│   ├── embeddings.py         # Embedding generation (sentence-transformers)
│   ├── vector_store.py       # FAISS operations
│   ├── llm_client.py         # LLM API integration (OpenAI/Groq)
│   ├── rag_pipeline.py       # RAG orchestration
│   └── logging_config.py     # Logging setup
├── data/                     # Uploaded documents (gitignored)
├── logs/                     # Application logs (gitignored)
├── faiss_index/              # Vector database (gitignored)
├── tests/
│   ├── __init__.py
│   └── test_document_processor.py
├── .env.example              # Environment template
├── .gitignore
├── requirements.txt          # Python dependencies
├── run.py                    # Application runner
└── README.md                 # This file
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `main.py` | Streamlit UI, user interactions |
| `document_processor.py` | File parsing, text chunking |
| `embeddings.py` | Vector embedding generation |
| `vector_store.py` | FAISS CRUD operations |
| `llm_client.py` | OpenAI/Groq API communication |
| `rag_pipeline.py` | End-to-end RAG orchestration |
| `config.py` | All configurable parameters |

---

## 🚧 Limitations & Future Improvements

### Current Limitations
- **Single-user**: No multi-user session management
- **No conversation memory**: Each question is independent
- **Basic chunking**: Fixed-size chunks (no semantic boundaries)
- **English-focused**: Embedding model optimized for English

### Future Improvements

#### Short-term
- [ ] **Streaming responses**: Show answer as it's generated
- [ ] **Conversation history**: Multi-turn conversations
- [ ] **Document preview**: View uploaded documents in UI
- [ ] **Better citations**: Highlight exact source passages

#### Medium-term
- [ ] **Hybrid search**: Combine semantic + keyword search
- [ ] **Smart chunking**: Use document structure (headings, paragraphs)
- [ ] **Multi-language**: Support multiple languages
- [ ] **Document categories**: Filter by document type/topic

#### Long-term
- [ ] **Deployment**: Docker + cloud deployment
- [ ] **Authentication**: User accounts and private documents
- [ ] **Analytics**: Track usage and popular questions
- [ ] **Fine-tuning**: Custom embedding models

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app
```

---

## 📄 License

MIT License - feel free to use this project for learning and development.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📞 Support

If you have questions or run into issues:
1. Check the logs in the `logs/` directory
2. Ensure your API keys are correctly configured
3. Open an issue on GitHub

---

**Built with ❤️ using Python, Streamlit, FAISS, and LLMs**
