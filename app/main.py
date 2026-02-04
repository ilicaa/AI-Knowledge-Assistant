"""
AI Knowledge Assistant - Streamlit Application

Main entry point for the Streamlit web interface.
Provides document upload, question answering, and document management.
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from app.config import SUPPORTED_EXTENSIONS, TOP_K_RESULTS
from app.document_processor import DocumentProcessor
from app.vector_store import get_vector_store
from app.rag_pipeline import get_rag_pipeline
from app.llm_client import get_llm_client


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F77B4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1F77B4;
    }
    .metric-box {
        background-color: #e8f4ea;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        display: inline-block;
        margin: 0.2rem;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #44ff44;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vector_store()
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = get_rag_pipeline()
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()


# =============================================================================
# SIDEBAR - DOCUMENT MANAGEMENT
# =============================================================================
def render_sidebar():
    """Render the sidebar with document management."""
    with st.sidebar:
        st.markdown("## 📁 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        
        # Process uploaded files
        if uploaded_files:
            process_button = st.button("📥 Process Documents", type="primary", use_container_width=True)
            
            if process_button:
                process_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Currently loaded documents
        st.markdown("### 📚 Loaded Documents")
        
        docs = st.session_state.vector_store.list_documents()
        
        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{doc['filename']}**")
                    st.caption(f"{doc['chunk_count']} chunks")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['filename']}", help=f"Delete {doc['filename']}"):
                        delete_document(doc['filename'])
                        st.rerun()
        else:
            st.info("No documents loaded yet. Upload some documents to get started!")
        
        st.divider()
        
        # Reset button
        if docs:
            if st.button("🔄 Reset All Documents", type="secondary", use_container_width=True):
                reset_all_documents()
                st.rerun()
        
        # System status
        st.markdown("### ⚙️ System Status")
        
        llm_client = get_llm_client()
        provider_info = llm_client.get_provider_info()
        
        if provider_info["configured"]:
            st.success(f"✅ LLM: {provider_info['provider'].upper()} ({provider_info['model']})")
        else:
            st.error("❌ LLM: Not configured (check .env file)")
        
        doc_count = st.session_state.vector_store.document_count
        st.info(f"📊 Total chunks: {doc_count}")


def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to vector store."""
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    successful = 0
    failed = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        try:
            # Read file content
            file_content = uploaded_file.read()
            
            # Process document
            processor = st.session_state.processor
            result = processor.process_file(uploaded_file.name, file_content)
            
            if result.status == "success":
                # Add to vector store
                if st.session_state.vector_store.add_document(result):
                    successful += 1
                    st.session_state.documents_loaded.append(uploaded_file.name)
                else:
                    failed += 1
            else:
                failed += 1
                logger.error(f"Failed to process {uploaded_file.name}: {result.error_message}")
            
        except Exception as e:
            failed += 1
            logger.error(f"Error processing {uploaded_file.name}: {e}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    if successful > 0:
        st.sidebar.success(f"✅ Successfully processed {successful} document(s)")
    if failed > 0:
        st.sidebar.error(f"❌ Failed to process {failed} document(s)")


def delete_document(filename: str):
    """Delete a document from the vector store."""
    if st.session_state.vector_store.delete_document(filename):
        if filename in st.session_state.documents_loaded:
            st.session_state.documents_loaded.remove(filename)
        st.sidebar.success(f"Deleted {filename}")
    else:
        st.sidebar.error(f"Failed to delete {filename}")


def reset_all_documents():
    """Reset all documents and clear the vector store."""
    st.session_state.vector_store.clear()
    st.session_state.documents_loaded = []
    st.session_state.messages = []
    st.sidebar.success("All documents cleared!")


# =============================================================================
# MAIN CONTENT - QUESTION ANSWERING
# =============================================================================
def render_main_content():
    """Render the main content area."""
    # Header
    st.markdown('<p class="main-header">🤖 AI Knowledge Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your uploaded documents</p>', unsafe_allow_html=True)
    
    # Check if ready
    status = st.session_state.rag_pipeline.is_ready()
    
    if not status["llm_configured"]:
        st.error("""
        ⚠️ **LLM not configured!**
        
        Please set up your API key:
        1. Copy `.env.example` to `.env`
        2. Add your OpenAI or Groq API key
        3. Restart the application
        """)
        return
    
    if not status["documents_loaded"]:
        st.warning("""
        📄 **No documents loaded!**
        
        Upload documents using the sidebar to get started.
        Supported formats: PDF, TXT, MD
        """)
    
    # Question input
    st.markdown("### 💬 Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="What would you like to know about your documents?",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True, disabled=not status["ready"])
    
    # Process question
    if ask_button and question:
        handle_question(question)
    
    # Display conversation history
    st.markdown("### 📝 Conversation History")
    
    if st.session_state.messages:
        for i, msg in enumerate(reversed(st.session_state.messages)):
            render_message(msg, len(st.session_state.messages) - i)
    else:
        st.info("No conversations yet. Ask a question to get started!")


def handle_question(question: str):
    """Handle a user question and generate response."""
    if not question.strip():
        st.warning("Please enter a question.")
        return
    
    # Show processing indicator
    with st.spinner("🔍 Searching documents and generating answer..."):
        start_time = time.time()
        
        # Get response from RAG pipeline
        response = st.session_state.rag_pipeline.ask(question)
        
        # Add to conversation history
        st.session_state.messages.append({
            "question": question,
            "response": response,
            "timestamp": time.strftime("%H:%M:%S")
        })
    
    # Force refresh to show new message
    st.rerun()


def render_message(msg: dict, index: int):
    """Render a single message in the conversation."""
    response = msg["response"]
    
    with st.container():
        # Question
        st.markdown(f"**🙋 Question:** {msg['question']}")
        
        # Answer
        if response.success:
            st.markdown(f"**🤖 Answer:**")
            st.markdown(response.answer)
            
            # Sources
            if response.sources:
                with st.expander(f"📚 Sources ({len(response.sources)} chunks used)", expanded=False):
                    for source in response.sources:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>📄 {source['filename']}</strong> 
                            (Chunk {source['chunk_index'] + 1}/{source['total_chunks']}) | 
                            Relevance: {source['relevance_score']:.1%}
                            <br><br>
                            <em>"{source['snippet']}"</em>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ Total: {response.total_time:.2f}s")
            with col2:
                st.caption(f"🔍 Retrieval: {response.retrieval_time:.3f}s")
            with col3:
                st.caption(f"🤖 Generation: {response.generation_time:.2f}s")
            
            # Performance warning
            if response.total_time > 5.0:
                st.warning(f"⚠️ Response time ({response.total_time:.2f}s) exceeded 5s target")
        else:
            st.markdown(f"""
            <div class="error-box">
                ❌ <strong>Error:</strong> {response.error_message}
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
