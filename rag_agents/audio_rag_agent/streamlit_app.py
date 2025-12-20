"""
Audio Query Agent - Streamlit UI
Agentic RAG System for Audio Question Answering via Transcription
"""

import streamlit as st
import asyncio
import os
from datetime import datetime
from agent import run_audio_processing, run_query, format_duration

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Audio Query Agent - RAG",
    page_icon="🎤",
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

    if "audio_info" not in st.session_state:
        st.session_state.audio_info = None

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
    has_assemblyai_key = bool(os.getenv("ASSEMBLYAI_API_KEY"))

    st.markdown(f"{'✅' if has_openai_key else '❌'} OpenAI API Key")
    st.markdown(f"{'✅' if has_assemblyai_key else '❌'} AssemblyAI API Key")

    if not has_openai_key or not has_assemblyai_key:
        st.warning("⚠️ Please set API keys in .env file")
        st.markdown("""
        **Required:**
        - `OPENAI_API_KEY` (for embeddings & GPT-4o)
        - `ASSEMBLYAI_API_KEY` (for transcription)
        """)

    st.markdown("---")

    # How It Works
    st.markdown("### 📋 How It Works")
    st.markdown("""
    1. **Upload** your audio file
    2. **AI transcribes** using AssemblyAI
    3. **Ask questions** about the content
    """)

    st.markdown("---")

    # Features
    st.markdown("### 🎯 Features")
    st.markdown("""
    🎤 Audio transcription
    🧠 Intelligent chunking
    🔍 Semantic search
    💡 Context-aware answers
    ⏱️ Timestamp citations
    """)

    st.markdown("---")

    # Audio Info
    if st.session_state.audio_info:
        st.markdown("### 🎤 Audio Info")
        st.markdown(f"**File:** {st.session_state.audio_info['filename']}")
        st.markdown(f"**Duration:** {st.session_state.audio_info['duration']}")
        st.markdown(f"**Words:** {st.session_state.audio_info['word_count']}")
        st.markdown(f"**Chunks:** {st.session_state.audio_info['chunk_count']}")
        st.markdown(f"**Processed:** {st.session_state.audio_info['timestamp']}")


# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>🎤 Audio Query Agent - Agentic RAG System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Upload audio → AI transcribes it → Ask questions → Get accurate answers with timestamp citations</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ============================================================================
# STAGE 1: AUDIO UPLOAD
# ============================================================================

if st.session_state.workflow_stage == "upload":
    st.markdown("## 🎤 Step 1: Upload Your Audio File")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'mp4', 'ogg', 'flac', 'webm'],
        help="Upload an audio file to transcribe and query (recommended max 30 minutes)"
    )

    if uploaded_file is not None:
        # Save to temp file
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file_path = temp_path

        # Display file info
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")

        # Cost estimate (rough approximation)
        if file_size_mb > 10:
            st.info(f"📊 Note: Transcription may take a few minutes for larger files. AssemblyAI charges ~$0.015/minute.")

        # Process Audio Button
        def start_processing():
            if not os.getenv("OPENAI_API_KEY") or not os.getenv("ASSEMBLYAI_API_KEY"):
                st.error("❌ API keys not found. Please check .env file")
                return
            st.session_state.is_processing = True
            st.session_state.current_filename = uploaded_file.name

        st.button(
            "🎤 Transcribe & Process Audio",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_processing,
            on_click=start_processing
        )

        # Processing
        if st.session_state.is_processing:
            with st.spinner("🔄 Processing your audio..."):
                with st.expander("📋 Processing Steps", expanded=True):
                    st.write("📤 Uploading to AssemblyAI...")
                    st.write("🎤 Transcribing speech...")
                    st.write("🧠 Chunking transcript...")
                    st.write("🔍 Generating embeddings...")
                    st.write("📊 Storing in vector database...")

                # Run audio processing
                result = st.session_state.loop.run_until_complete(
                    run_audio_processing(
                        audio_path=st.session_state.uploaded_file_path,
                        filename=uploaded_file.name
                    )
                )

            # Check for errors
            if result.get("errors"):
                st.error(f"❌ Error processing audio: {result['errors'][0]}")
                st.session_state.is_processing = False
            else:
                # Store results
                st.session_state.vector_store = result["vector_store"]
                st.session_state.audio_info = {
                    "filename": uploaded_file.name,
                    "duration": format_duration(result["audio_duration"]),
                    "word_count": result["word_count"],
                    "chunk_count": result["chunk_count"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Success message
                st.success("✅ Audio processed successfully!")
                st.info(f"📊 **Statistics:**\n- Duration: {format_duration(result['audio_duration'])}\n- Words: {result['word_count']}\n- Chunks: {result['chunk_count']}")

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
                <li><strong>Upload:</strong> Select an audio file using the uploader above</li>
                <li><strong>Process:</strong> Click "Transcribe & Process Audio" to transcribe and index</li>
                <li><strong>Query:</strong> Ask questions about what was said in the audio</li>
            </ol>
            <h4>Supported Formats:</h4>
            <p>MP3, WAV, M4A, MP4, OGG, FLAC, WEBM</p>
            <h4>Example Use Cases:</h4>
            <p>
            • Podcast episodes and interviews<br>
            • Recorded meetings and lectures<br>
            • Voice memos and notes<br>
            • Audio books and tutorials
            </p>
            <h4>Example Questions:</h4>
            <p>
            • "What topics were discussed?"<br>
            • "Summarize the main points"<br>
            • "What did they say about [specific topic]?"
            </p>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# STAGE 2: QUERY INTERFACE
# ============================================================================

elif st.session_state.workflow_stage == "query":
    st.markdown("## 🔍 Step 2: Ask Questions About Your Audio")

    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="What topics were discussed in this audio?",
        help="Ask any question about the audio content"
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
        def reset_audio():
            st.session_state.workflow_stage = "upload"
            st.session_state.vector_store = None
            st.session_state.audio_info = None
            st.session_state.query_history = []
            st.session_state.uploaded_file_path = None

        st.button(
            "🔄 New Audio",
            type="secondary",
            use_container_width=True,
            on_click=reset_audio
        )

    # Process query
    if st.session_state.is_processing:
        with st.spinner("🤖 Searching transcript and generating answer..."):
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
            st.markdown("### ⏱️ Timestamp Sources")
            if latest["citations"]:
                for citation in latest["citations"]:
                    st.info(citation)
            else:
                st.warning("No sources found")

        # Retrieved context chunks (expandable)
        with st.expander("📋 Retrieved Transcript Segments"):
            for i, chunk in enumerate(latest["retrieved_chunks"]):
                timestamp_range = chunk["metadata"].get("timestamp_range", "?")
                st.markdown(f"**Segment {i+1} (at {timestamp_range}):**")
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
                <li><strong>Be specific:</strong> Ask focused questions about particular topics or moments</li>
                <li><strong>Ask follow-ups:</strong> Build on previous questions for deeper insights</li>
                <li><strong>Reference context:</strong> Mention topics or themes you heard</li>
            </ul>
            <h4>Example Questions:</h4>
            <p>
            ✓ "What were the main topics discussed?"<br>
            ✓ "Summarize what was said about [topic]"<br>
            ✓ "What recommendations were made?"<br>
            ✓ "What points were emphasized?"
            </p>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Tech Stack:</strong> LangGraph • GPT-4o • AssemblyAI • ChromaDB • Streamlit</p>
    <p>Built with ❤️ for intelligent audio question answering</p>
</div>
""", unsafe_allow_html=True)
