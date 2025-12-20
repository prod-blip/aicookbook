"""
Audio RAG Agent - LangGraph Implementation
Processes audio files via AssemblyAI transcription and enables Q&A using Retrieval Augmented Generation
"""

import os
import operator
import asyncio
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions

# AssemblyAI import
import assemblyai as aai

# Environment
from dotenv import load_dotenv
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 4
LLM_TEMPERATURE = 0


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AudioRAGAgentState(TypedDict):
    """State for Audio RAG Agent workflow"""

    # Audio processing inputs
    audio_path: str
    filename: str
    mode: str  # "process_audio" or "query"

    # Transcription outputs
    transcript_text: Optional[str]
    transcript_id: Optional[str]
    audio_duration: Optional[float]  # in seconds
    word_count: Optional[int]

    # Chunking outputs
    chunks: Optional[List[Dict]]
    chunk_count: Optional[int]

    # Embedding outputs
    vector_store: Optional[Any]  # ChromaDB collection
    embedding_status: Optional[str]

    # Query inputs
    query: str

    # Query outputs
    retrieved_chunks: Optional[List[Dict]]
    sources: Optional[List[Dict]]
    answer: Optional[str]
    citations: Optional[List[str]]

    # Error tracking
    messages: Annotated[list, operator.add]
    errors: Annotated[List[str], operator.add]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (MM:SS)
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """
    Convert seconds to human-readable duration.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "5m 30s" or "1h 15m")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# ============================================================================
# NODE 1: TRANSCRIBE AUDIO
# ============================================================================

async def transcribe_audio(state: AudioRAGAgentState) -> AudioRAGAgentState:
    """
    Transcribe audio file using AssemblyAI.

    Reads: state["audio_path"], state["filename"]
    Updates: state["transcript_text"], state["audio_duration"],
             state["word_count"], state["transcript_id"]
    """
    print("\n=== 🎤 Transcribing Audio ===")

    try:
        audio_path = state["audio_path"]

        # Check API key
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise Exception("ASSEMBLYAI_API_KEY not found in environment")

        # Initialize AssemblyAI client
        aai.settings.api_key = api_key

        # Configure transcriber (no speaker diarization per requirements)
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            language_code="en",
            speaker_labels=False,
            punctuate=True,
            format_text=True
        )

        transcriber = aai.Transcriber(config=config)

        print(f"📤 Uploading audio file: {state['filename']}")

        # Submit transcription
        transcript = transcriber.transcribe(audio_path)

        # Poll for completion
        print("⏳ Transcription in progress...")
        while transcript.status not in [
            aai.TranscriptStatus.completed,
            aai.TranscriptStatus.error
        ]:
            await asyncio.sleep(3)  # Check every 3 seconds
            # Note: In production, you'd call transcriber.wait_for_completion()
            # but we use polling for better async control

        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")

        # Extract transcript details
        transcript_text = transcript.text
        audio_duration = transcript.audio_duration / 1000  # ms to seconds
        word_count = len(transcript_text.split())

        print(f"✅ Transcription complete:")
        print(f"   Duration: {format_duration(audio_duration)}")
        print(f"   Words: {word_count}")
        print(f"   Characters: {len(transcript_text)}")

        return {
            **state,
            "transcript_text": transcript_text,
            "transcript_id": transcript.id,
            "audio_duration": audio_duration,
            "word_count": word_count,
        }

    except Exception as e:
        print(f"❌ Error transcribing audio: {e}")
        return {
            **state,
            "transcript_text": None,
            "transcript_id": None,
            "audio_duration": 0,
            "word_count": 0,
            "errors": [f"Audio transcription failed: {str(e)}"]
        }


# ============================================================================
# NODE 2: CHUNK TEXT
# ============================================================================

async def chunk_text(state: AudioRAGAgentState) -> AudioRAGAgentState:
    """
    Split transcript into chunks with timestamp metadata.

    Reads: state["transcript_text"], state["filename"], state["audio_duration"]
    Updates: state["chunks"], state["chunk_count"]
    """
    print("\n=== 🧠 Chunking Transcript ===")

    try:
        transcript_text = state["transcript_text"]
        filename = state["filename"]
        audio_duration = state["audio_duration"]

        if not transcript_text:
            print("⏭️ No transcript to chunk, skipping")
            return {**state}

        # Create text splitter (same as PDF RAG)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        # Split text into chunks
        text_chunks = text_splitter.split_text(transcript_text)

        # Calculate approximate timestamp ranges using linear interpolation
        total_chars = len(transcript_text)
        chunks = []

        for i, chunk_text in enumerate(text_chunks):
            # Estimate timestamp based on character position
            chunk_start_char = sum(len(c) for c in text_chunks[:i])
            chunk_end_char = chunk_start_char + len(chunk_text)

            # Linear interpolation for timestamps
            start_time = (chunk_start_char / total_chars) * audio_duration if total_chars > 0 else 0
            end_time = (chunk_end_char / total_chars) * audio_duration if total_chars > 0 else 0

            chunk_dict = {
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "start_time": start_time,
                    "end_time": end_time,
                    "timestamp_range": f"{format_timestamp(start_time)} - {format_timestamp(end_time)}",
                    "chunk_index": i,
                    "upload_timestamp": datetime.now().isoformat()
                }
            }
            chunks.append(chunk_dict)

        chunk_count = len(chunks)
        print(f"✅ Created {chunk_count} chunks ({CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)")

        return {
            **state,
            "chunks": chunks,
            "chunk_count": chunk_count,
        }

    except Exception as e:
        print(f"❌ Error chunking text: {e}")
        return {
            **state,
            "chunks": None,
            "chunk_count": 0,
            "errors": [f"Text chunking failed: {str(e)}"]
        }


# ============================================================================
# NODE 3: GENERATE EMBEDDINGS
# ============================================================================

async def generate_embeddings(state: AudioRAGAgentState) -> AudioRAGAgentState:
    """
    Create embeddings and store in ChromaDB.

    Reads: state["chunks"]
    Updates: state["vector_store"], state["embedding_status"]
    """
    print("\n=== 🔍 Generating Embeddings ===")

    try:
        chunks = state["chunks"]

        if not chunks:
            print("⏭️ No chunks to embed, skipping")
            return {**state}

        # Initialize ChromaDB client (in-memory)
        client = chromadb.Client()

        # Create OpenAI embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        # Create collection with timestamp
        collection_name = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        collection = client.create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )

        # Prepare documents and metadata
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Add to ChromaDB
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Embedded {len(chunks)} chunks into ChromaDB collection: {collection_name}")

        return {
            **state,
            "vector_store": collection,
            "embedding_status": f"Successfully embedded {len(chunks)} chunks",
        }

    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return {
            **state,
            "vector_store": None,
            "embedding_status": f"Error: {str(e)}",
            "errors": [f"Embedding generation failed: {str(e)}"]
        }


# ============================================================================
# NODE 4: PROCESS QUERY
# ============================================================================

async def process_query(state: AudioRAGAgentState) -> AudioRAGAgentState:
    """
    Retrieve relevant chunks for user query.

    Reads: state["query"], state["vector_store"]
    Updates: state["retrieved_chunks"], state["sources"]
    """
    print("\n=== 🔎 Processing Query ===")

    try:
        query = state["query"]
        vector_store = state["vector_store"]

        if not vector_store:
            print("❌ No vector store available")
            return {
                **state,
                "retrieved_chunks": None,
                "sources": None,
                "errors": ["No audio has been processed yet"]
            }

        print(f"📝 Query: {query}")

        # Query ChromaDB for similar chunks
        results = vector_store.query(
            query_texts=[query],
            n_results=TOP_K_CHUNKS
        )

        # Extract retrieved chunks with metadata
        retrieved_chunks = []
        sources = []

        if results and results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                chunk_info = {
                    "text": doc,
                    "metadata": metadata
                }
                retrieved_chunks.append(chunk_info)

                # Create source citation
                source = {
                    "filename": metadata.get("filename", "Unknown"),
                    "timestamp_range": metadata.get("timestamp_range", "?"),
                    "chunk_index": metadata.get("chunk_index", i),
                    "metadata": metadata
                }
                sources.append(source)

        print(f"✅ Retrieved {len(retrieved_chunks)} relevant chunks")

        return {
            **state,
            "retrieved_chunks": retrieved_chunks,
            "sources": sources,
        }

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return {
            **state,
            "retrieved_chunks": None,
            "sources": None,
            "errors": [f"Query processing failed: {str(e)}"]
        }


# ============================================================================
# NODE 5: GENERATE ANSWER
# ============================================================================

async def generate_answer(state: AudioRAGAgentState) -> AudioRAGAgentState:
    """
    Generate answer using LLM with retrieved context.

    Reads: state["query"], state["retrieved_chunks"], state["sources"]
    Updates: state["answer"], state["citations"]
    """
    print("\n=== 💡 Generating Answer ===")

    try:
        query = state["query"]
        retrieved_chunks = state["retrieved_chunks"]
        sources = state["sources"]

        if not retrieved_chunks:
            print("❌ No retrieved chunks available")
            return {
                **state,
                "answer": "I couldn't find relevant information to answer your question.",
                "citations": []
            }

        # Build context from retrieved chunks with timestamps
        context = ""
        for i, chunk in enumerate(retrieved_chunks):
            timestamp_range = chunk["metadata"].get("timestamp_range", "?")
            context += f"\n[Source {i+1}, Timestamp: {timestamp_range}]\n{chunk['text']}\n"

        filename = sources[0]["filename"] if sources else "audio"

        # Create RAG prompt (adapted for audio)
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context from an audio transcript.

Context from audio transcript ({filename}):
{context}

User Question: {query}

Instructions:
- Answer the question based ONLY on the provided transcript context
- If the context doesn't contain enough information, say so clearly
- Cite specific sources using the format: (at MM:SS)
- Be concise but thorough
- Use markdown formatting for readability
- Reference the audio content naturally (e.g., "The speaker mentioned at 03:45...")

Answer:"""

        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=LLM_TEMPERATURE
        )

        # Generate answer
        print("🤖 Asking GPT-4o...")
        response = await llm.ainvoke(prompt)
        answer = response.content

        # Create citations list with timestamps
        citations = []
        for source in sources:
            timestamp_range = source.get("timestamp_range", "?")
            citation = f"{source['filename']} (at {timestamp_range})"
            if citation not in citations:
                citations.append(citation)

        print(f"✅ Generated answer ({len(answer)} characters)")

        return {
            **state,
            "answer": answer,
            "citations": citations,
        }

    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return {
            **state,
            "answer": f"Error generating answer: {str(e)}",
            "citations": [],
            "errors": [f"Answer generation failed: {str(e)}"]
        }


# ============================================================================
# ROUTING FUNCTION
# ============================================================================

def route_mode(state: AudioRAGAgentState) -> str:
    """
    Route based on mode in state.

    Returns node name to execute next.
    """
    mode = state.get("mode", "query")

    if mode == "process_audio":
        return "transcribe_audio"
    else:  # mode == "query"
        return "process_query"


# ============================================================================
# GRAPH CREATION
# ============================================================================

def create_graph():
    """Create and compile the LangGraph workflow"""

    # Initialize graph
    workflow = StateGraph(AudioRAGAgentState)

    # Add nodes
    workflow.add_node("transcribe_audio", transcribe_audio)
    workflow.add_node("chunk_text", chunk_text)
    workflow.add_node("generate_embeddings", generate_embeddings)
    workflow.add_node("process_query", process_query)
    workflow.add_node("generate_answer", generate_answer)

    # Set conditional entry point
    workflow.set_conditional_entry_point(
        route_mode,
        {
            "transcribe_audio": "transcribe_audio",
            "process_query": "process_query"
        }
    )

    # Add edges for audio processing flow
    workflow.add_edge("transcribe_audio", "chunk_text")
    workflow.add_edge("chunk_text", "generate_embeddings")
    workflow.add_edge("generate_embeddings", END)

    # Add edges for query flow
    workflow.add_edge("process_query", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Compile graph
    return workflow.compile()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def run_audio_processing(audio_path: str, filename: str):
    """
    Process an audio file through the agent.

    Args:
        audio_path: Path to audio file
        filename: Name of the audio file

    Returns:
        Dictionary with processing results
    """
    print("\n" + "="*60)
    print("🎤 AUDIO RAG AGENT - PROCESSING")
    print("="*60)

    graph = create_graph()

    initial_state = {
        "audio_path": audio_path,
        "filename": filename,
        "mode": "process_audio",
        "transcript_text": None,
        "transcript_id": None,
        "audio_duration": None,
        "word_count": None,
        "chunks": None,
        "chunk_count": None,
        "vector_store": None,
        "embedding_status": None,
        "query": "",
        "retrieved_chunks": None,
        "sources": None,
        "answer": None,
        "citations": None,
        "messages": [],
        "errors": []
    }

    result = await graph.ainvoke(initial_state)

    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE")
    print("="*60)

    return {
        "audio_duration": result.get("audio_duration", 0),
        "word_count": result.get("word_count", 0),
        "chunk_count": result.get("chunk_count", 0),
        "vector_store": result.get("vector_store"),
        "embedding_status": result.get("embedding_status", ""),
        "errors": result.get("errors", [])
    }


async def run_query(query: str, vector_store):
    """
    Query the processed audio transcript.

    Args:
        query: User question
        vector_store: ChromaDB collection

    Returns:
        Dictionary with answer and sources
    """
    graph = create_graph()

    initial_state = {
        "audio_path": "",
        "filename": "",
        "mode": "query",
        "transcript_text": None,
        "transcript_id": None,
        "audio_duration": None,
        "word_count": None,
        "chunks": None,
        "chunk_count": None,
        "vector_store": vector_store,
        "embedding_status": None,
        "query": query,
        "retrieved_chunks": None,
        "sources": None,
        "answer": None,
        "citations": None,
        "messages": [],
        "errors": []
    }

    result = await graph.ainvoke(initial_state)

    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "sources": result.get("sources", []),
        "errors": result.get("errors", [])
    }
