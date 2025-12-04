"""
PDF Query Agent - Streamlit UI
Agentic RAG System for PDF Question Answering
"""

import streamlit as st
import asyncio
import os
from datetime import datetime
from agent import run_document_processing, run_query

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="PDF Query Agent - RAG",
    page_icon="📚",
    layout="wide"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""

    # Event loop - ONLY ONCE
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)

    if "workflow_stage" not in st.session_state:
        st.session_state.workflow_stage = "upload"

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "document_info" not in st.session_state:
        st.session_state.document_info = None

    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None


init_session_state()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 🔑 Configuration Status")

    # API Key Status
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    st.markdown(f"{'✅' if has_openai_key else '❌'} OpenAI API Key")

    if not has_openai_key:
        st.warning("⚠️ Please set OPENAI_API_KEY in .env file")

    st.markdown("---")

    # How It Works
    st.markdown("### 📋 How It Works")
    st.markdown("""
    1. **Upload** your PDF document
    2. **AI processes** and indexes content
    3. **Ask questions** and get answers
    """)

    st.markdown("---")

    # Features
    st.markdown("### 🎯 Features")
    st.markdown("""
    📄 PDF text extraction
    🧠 Intelligent chunking
    🔍 Semantic search
    💡 Context-aware answers
    📌 Source citations
    """)

    st.markdown("---")

    # Document Info
    if st.session_state.document_info:
        st.markdown("### 📄 Document Info")
        st.markdown(f"**File:** {st.session_state.document_info['filename']}")
        st.markdown(f"**Pages:** {st.session_state.document_info['page_count']}")
        st.markdown(f"**Chunks:** {st.session_state.document_info['chunk_count']}")
        st.markdown(f"**Processed:** {st.session_state.document_info['timestamp']}")


# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>📚 PDF Query Agent - Agentic RAG System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Upload a PDF → AI processes it → Ask questions → Get accurate answers with source citations</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ============================================================================
# STAGE 1: DOCUMENT UPLOAD
# ============================================================================

if st.session_state.workflow_stage == "upload":
    st.markdown("## 📄 Step 1: Upload Your PDF Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to start querying"
    )

    if uploaded_file is not None:
        # Save to temp file
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file_path = temp_path
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Process Document Button
        def start_processing():
            st.session_state.is_processing = True
            st.session_state.current_filename = uploaded_file.name

        st.button(
            "📄 Process Document",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_processing,
            on_click=start_processing
        )

        # Processing
        if st.session_state.is_processing:
            with st.spinner("🔄 Processing your PDF..."):
                with st.expander("📋 Processing Steps", expanded=True):
                    st.write("⏳ Loading PDF...")
                    st.write("🧠 Chunking text...")
                    st.write("🔍 Generating embeddings...")
                    st.write("📊 Storing in vector database...")

                # Run document processing
                result = st.session_state.loop.run_until_complete(
                    run_document_processing(
                        pdf_path=st.session_state.uploaded_file_path,
                        filename=uploaded_file.name
                    )
                )

            # Check for errors
            if result.get("errors"):
                st.error(f"❌ Error processing document: {result['errors'][0]}")
                st.session_state.is_processing = False
            else:
                # Store results
                st.session_state.vector_store = result["vector_store"]
                st.session_state.document_info = {
                    "filename": uploaded_file.name,
                    "page_count": result["page_count"],
                    "chunk_count": result["chunk_count"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Success message
                st.success("✅ Document processed successfully!")
                st.info(f"📊 **Statistics:**\n- Pages: {result['page_count']}\n- Chunks: {result['chunk_count']}")

                # Transition to query stage
                st.session_state.workflow_stage = "query"
                st.session_state.is_processing = False
                st.rerun()

    else:
        # Help section when no file uploaded
        st.markdown("---")
        st.markdown("""
            <div style='padding: 25px; background-color: #f0f2f6; border-radius: 15px;'>
            <h3>🚀 Getting Started</h3>
            <ol>
                <li><strong>Upload:</strong> Select a PDF file using the uploader above</li>
                <li><strong>Process:</strong> Click "Process Document" to index your PDF</li>
                <li><strong>Query:</strong> Ask questions about the document content</li>
            </ol>
            <h4>Example Questions:</h4>
            <p>
            • "What are the main topics covered in this document?"<br>
            • "Summarize the key findings from section 3"<br>
            • "What does the author say about [specific topic]?"
            </p>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# STAGE 2: QUERY INTERFACE
# ============================================================================

elif st.session_state.workflow_stage == "query":
    st.markdown("## 🔍 Step 2: Ask Questions About Your Document")

    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?",
        help="Ask any question about the PDF content"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        def start_query():
            if not query.strip():
                st.error("❌ Please enter a question")
                return
            st.session_state.is_processing = True
            st.session_state.current_query = query

        st.button(
            "🔍 Ask Question",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_processing,
            on_click=start_query
        )

    with col2:
        def reset_document():
            st.session_state.workflow_stage = "upload"
            st.session_state.vector_store = None
            st.session_state.document_info = None
            st.session_state.query_history = []
            st.session_state.uploaded_file_path = None

        st.button(
            "🔄 New Document",
            type="secondary",
            use_container_width=True,
            on_click=reset_document
        )

    # Process query
    if st.session_state.is_processing:
        with st.spinner("🤖 Searching document and generating answer..."):
            # Run query
            result = st.session_state.loop.run_until_complete(
                run_query(
                    query=st.session_state.current_query,
                    vector_store=st.session_state.vector_store
                )
            )

        # Check for errors
        if result.get("errors"):
            st.error(f"❌ Error: {result['errors'][0]}")
        else:
            # Add to query history
            st.session_state.query_history.append({
                "query": st.session_state.current_query,
                "answer": result["answer"],
                "citations": result["citations"],
                "retrieved_chunks": result["retrieved_chunks"],
                "sources": result["sources"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        st.session_state.is_processing = False
        st.rerun()

    # Display results
    if st.session_state.query_history:
        st.markdown("---")

        # Get latest Q&A
        latest = st.session_state.query_history[-1]

        # Two-column layout
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("### 💡 Answer")
            st.markdown(latest["answer"])

        with col_right:
            st.markdown("### 📌 Sources")
            if latest["citations"]:
                for citation in latest["citations"]:
                    st.info(citation)
            else:
                st.warning("No sources found")

        # Retrieved context chunks (expandable)
        with st.expander("📋 Retrieved Context Chunks"):
            for i, chunk in enumerate(latest["retrieved_chunks"]):
                page = chunk["metadata"].get("page_number", "?")
                st.markdown(f"**Chunk {i+1} (Page {page}):**")
                st.text(chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"])
                st.markdown("---")

        # Query history (expandable)
        if len(st.session_state.query_history) > 1:
            with st.expander(f"📜 Query History ({len(st.session_state.query_history)} questions)"):
                for i, qa in enumerate(reversed(st.session_state.query_history[:-1])):
                    st.markdown(f"**Q{len(st.session_state.query_history) - i - 1}: {qa['query']}**")
                    st.markdown(f"*{qa['timestamp']}*")
                    st.markdown(qa['answer'][:200] + "..." if len(qa['answer']) > 200 else qa['answer'])
                    st.markdown("---")

    else:
        # Help section for first query
        st.markdown("---")
        st.markdown("""
            <div style='padding: 25px; background-color: #f0f2f6; border-radius: 15px;'>
            <h3>💡 Tips for Better Results</h3>
            <ul>
                <li><strong>Be specific:</strong> Ask focused questions about particular topics</li>
                <li><strong>Reference context:</strong> Mention sections, chapters, or page numbers if known</li>
                <li><strong>Follow up:</strong> Build on previous questions for deeper insights</li>
            </ul>
            <h4>Example Questions:</h4>
            <p>
            ✓ "What methodology was used in this research?"<br>
            ✓ "Summarize the conclusions on page 5"<br>
            ✓ "What are the key recommendations?"
            </p>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ using LangGraph, GPT-4o, ChromaDB & Streamlit</p>",
    unsafe_allow_html=True
)
