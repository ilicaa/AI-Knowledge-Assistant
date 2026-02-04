"""
Configuration settings for the AI Knowledge Assistant.
Centralizes all configurable parameters for easy tuning.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
FAISS_DIR = BASE_DIR / "faiss_index"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)

# =============================================================================
# DOCUMENT PROCESSING CONFIGURATION
# =============================================================================
# Chunk size in characters (not tokens) - optimized for performance
CHUNK_SIZE = 500
# Overlap between chunks to maintain context
CHUNK_OVERLAP = 50
# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
# Using a lightweight, fast model for embeddings
# all-MiniLM-L6-v2 is a good balance of speed and quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Dimension of the embedding vectors (384 for MiniLM)
EMBEDDING_DIMENSION = 384

# =============================================================================
# VECTOR STORE CONFIGURATION
# =============================================================================
# FAISS index collection name
COLLECTION_NAME = "documents"
# Number of results to retrieve (top-k)
TOP_K_RESULTS = 5
# Similarity threshold (optional filtering)
SIMILARITY_THRESHOLD = 0.3

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# OpenAI API Key (alternative, no api key provided currently because Groq is fine for now)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM Provider: "openai" or "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Model names
GROQ_MODEL = "openai/gpt-oss-20b"  
OPENAI_MODEL = "gpt-3.5-turbo"

# LLM Parameters
MAX_TOKENS = 512
TEMPERATURE = 0.3  # Lower for more factual responses
REQUEST_TIMEOUT = 30  # seconds

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================
# Maximum context length to send to LLM (in characters)
MAX_CONTEXT_LENGTH = 3000
# Cache embeddings to avoid recomputation
CACHE_EMBEDDINGS = True
# Inference timeout warning threshold (seconds)
INFERENCE_WARNING_THRESHOLD = 5.0

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "app.log"
