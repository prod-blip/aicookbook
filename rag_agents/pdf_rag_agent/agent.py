"""
PDF RAG Agent - LangGraph Implementation
Processes PDF documents and enables Q&A using Retrieval Augmented Generation
"""

import os
import operator
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions

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

class RAGAgentState(TypedDict):
    """State for PDF RAG Agent workflow"""

    # Document processing inputs
    pdf_path: str
    filename: str
    mode: str  # "process_document" or "query"

    # Document processing outputs
    raw_text: Optional[str]
    page_count: Optional[int]
    chunks: Optional[List[Dict]]
    chunk_count: Optional[int]
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
# NODE 1: LOAD PDF
# ============================================================================

async def load_pdf(state: RAGAgentState) -> RAGAgentState:
    """
    Load and parse PDF document.

    Reads: state["pdf_path"]
    Updates: state["raw_text"], state["page_count"]
    """
    print("\n=== 📄 Loading PDF ===")

    try:
        pdf_path = state["pdf_path"]

        # Load PDF using LangChain's PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Extract text and page information
        raw_text = ""
        page_count = len(pages)

        for i, page in enumerate(pages):
            raw_text += f"\n[Page {i+1}]\n{page.page_content}\n"

        print(f"✅ Loaded PDF: {page_count} pages, {len(raw_text)} characters")

        return {
            **state,
            "raw_text": raw_text,
            "page_count": page_count,
        }

    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return {
            **state,
            "raw_text": None,
            "page_count": 0,
            "errors": [f"PDF loading failed: {str(e)}"]
        }


# ============================================================================
# NODE 2: CHUNK TEXT
# ============================================================================

async def chunk_text(state: RAGAgentState) -> RAGAgentState:
    """
    Split document into chunks with metadata.

    Reads: state["raw_text"], state["filename"]
    Updates: state["chunks"], state["chunk_count"]
    """
    print("\n=== 🧠 Chunking Text ===")

    try:
        raw_text = state["raw_text"]
        filename = state["filename"]

        if not raw_text:
            print("⏭️ No text to chunk, skipping")
            return {**state}

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        # Split text into chunks
        text_chunks = text_splitter.split_text(raw_text)

        # Create chunks with metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Extract page number from chunk (look for [Page X] marker)
            page_num = 1  # Default
            if "[Page " in chunk_text:
                try:
                    page_start = chunk_text.find("[Page ") + 6
                    page_end = chunk_text.find("]", page_start)
                    page_num = int(chunk_text[page_start:page_end])
                except:
                    pass

            chunk_dict = {
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "page_number": page_num,
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

async def generate_embeddings(state: RAGAgentState) -> RAGAgentState:
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
        collection_name = f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

async def process_query(state: RAGAgentState) -> RAGAgentState:
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
                "errors": ["No document has been processed yet"]
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
                    "page": metadata.get("page_number", "?"),
                    "chunk_index": metadata.get("chunk_index", i)
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

async def generate_answer(state: RAGAgentState) -> RAGAgentState:
    """
    Generate answer using LLM with retrieved context.

    Reads: state["query"], state["retrieved_chunks"]
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

        # Build context from retrieved chunks
        context = ""
        for i, chunk in enumerate(retrieved_chunks):
            page = chunk["metadata"].get("page_number", "?")
            context += f"\n[Source {i+1}, Page {page}]\n{chunk['text']}\n"

        filename = sources[0]["filename"] if sources else "document"

        # Create RAG prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context from a PDF document.

Context from PDF ({filename}):
{context}

User Question: {query}

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite specific sources using the format: (Page X)
- Be concise but thorough
- Use markdown formatting for readability

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

        # Create citations list
        citations = []
        for source in sources:
            citation = f"{source['filename']} (Page {source['page']})"
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

def route_mode(state: RAGAgentState) -> str:
    """
    Route based on mode in state.

    Returns node name to execute next.
    """
    mode = state.get("mode", "query")

    if mode == "process_document":
        return "load_pdf"
    else:  # mode == "query"
        return "process_query"


# ============================================================================
# GRAPH CREATION
# ============================================================================

def create_graph():
    """Create and compile the LangGraph workflow"""

    # Initialize graph
    workflow = StateGraph(RAGAgentState)

    # Add nodes
    workflow.add_node("load_pdf", load_pdf)
    workflow.add_node("chunk_text", chunk_text)
    workflow.add_node("generate_embeddings", generate_embeddings)
    workflow.add_node("process_query", process_query)
    workflow.add_node("generate_answer", generate_answer)

    # Set conditional entry point
    workflow.set_conditional_entry_point(
        route_mode,
        {
            "load_pdf": "load_pdf",
            "process_query": "process_query"
        }
    )

    # Add edges for document processing flow
    workflow.add_edge("load_pdf", "chunk_text")
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

async def run_document_processing(pdf_path: str, filename: str):
    """
    Process a PDF document through the agent.

    Args:
        pdf_path: Path to PDF file
        filename: Name of the PDF file

    Returns:
        Dictionary with processing results
    """
    graph = create_graph()

    initial_state = {
        "pdf_path": pdf_path,
        "filename": filename,
        "mode": "process_document",
        "raw_text": None,
        "page_count": None,
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

    return {
        "page_count": result.get("page_count", 0),
        "chunk_count": result.get("chunk_count", 0),
        "vector_store": result.get("vector_store"),
        "embedding_status": result.get("embedding_status", ""),
        "errors": result.get("errors", [])
    }


async def run_query(query: str, vector_store):
    """
    Query the processed document.

    Args:
        query: User question
        vector_store: ChromaDB collection

    Returns:
        Dictionary with answer and sources
    """
    graph = create_graph()

    initial_state = {
        "pdf_path": "",
        "filename": "",
        "mode": "query",
        "raw_text": None,
        "page_count": None,
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
