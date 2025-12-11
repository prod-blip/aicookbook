# PDF Query Agent - Agentic RAG System

An intelligent document Q&A assistant that processes PDF documents and answers questions using Retrieval Augmented Generation (RAG), featuring semantic search with ChromaDB and context-aware answers with source citations.

✨ **Powered by LangGraph agentic architecture with GPT-4o, OpenAI embeddings, and ChromaDB vector store**



https://github.com/user-attachments/assets/53bb9d5b-9dcb-4d1c-b1e4-7a6f00a94a61



## Features

* **PDF Upload & Processing** - Upload any PDF document and get it automatically chunked, embedded, and indexed for intelligent querying
* **Semantic Search** - Find relevant information using AI-powered similarity search across your document chunks
* **Context-Aware Answers** - Get accurate, detailed answers based only on your document content, not generic knowledge
* **Source Citations** - Every answer includes page numbers and source references for easy verification
* **Query History** - Track all your questions and answers within a session for easy reference
* **Intelligent Chunking** - Smart text splitting with overlap ensures context preservation across chunk boundaries

## Setup

### Requirements

* Python 3.8+
* OpenAI API Key (for GPT-4o and embeddings)
* Internet connection (for API access)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/prod-blip/aicookbook.git
cd aicookbook/rag_agents/pdf_rag_agent
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Get your API credentials:
   * **OpenAI API Key**: https://platform.openai.com/api-keys
     - Sign up or log in to OpenAI
     - Navigate to API Keys section
     - Create a new secret key
     - Copy and save securely

4. Setup your `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-proj-...your-actual-key-here
```

**Important:** Never commit the `.env` file to version control. It's included in `.gitignore`.

## Running the App

1. Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

2. Your browser will automatically open to `http://localhost:8501`

3. If the browser doesn't open automatically, manually navigate to the URL shown in the terminal

4. You'll see the two-stage workflow interface ready to use

## How It Works - Complete Workflow

### Stage 1: Upload PDF Document

1. Click "Choose a PDF file" to upload your document
2. Click "📄 Process Document" to start processing
3. The AI will:
   - Extract text from all pages
   - Split into 1000-character chunks with 200-character overlap
   - Generate embeddings using OpenAI text-embedding-3-small
   - Store in ChromaDB vector database
4. You'll see statistics: page count and chunk count
5. Automatically transitions to query stage

### Stage 2: Ask Questions

1. Enter your question in the text input
2. Click "🔍 Ask Question"
3. The AI will:
   - Search for the 4 most relevant chunks
   - Send them to GPT-4o with your question
   - Generate a comprehensive answer
   - Include source citations with page numbers
4. View your answer with sources
5. Ask follow-up questions or start over with a new document

**What you get:**
- Natural language answer based on document content
- Source citations showing which pages were referenced
- Retrieved context chunks for verification
- Query history for the session

**Actions available:**
- View retrieved context chunks
- Browse query history
- Upload new document (resets session)

## Agent Architecture

The application uses **LangGraph** with a **single conditional graph** and **5 specialized nodes**:

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  PDF QUERY AGENT WORKFLOW                   │
└─────────────────────────────────────────────────────────────┘

                      ┌────────────────┐
                      │  User Action   │
                      │ (mode selector)│
                      └────────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │  CONDITIONAL ROUTER │
                    │  route_mode()       │
                    └──┬────────────────┬─┘
                       │                │
       mode="process_document"    mode="query"
                       │                │
                       ▼                ▼

┌──────────────────────────────┐  ┌──────────────────────────────┐
│   DOCUMENT PROCESSING FLOW   │  │      QUERY FLOW              │
└──────────────────────────────┘  └──────────────────────────────┘

        ▼                                  ▼
╔═══════════════════════╗         ╔═══════════════════════╗
║  NODE 1: load_pdf     ║         ║ NODE 4: process_query ║
╠═══════════════════════╣         ╠═══════════════════════╣
║ • Load PDF file       ║         ║ • Query vector store  ║
║ • Extract text        ║         ║ • Similarity search   ║
║ • Parse pages         ║         ║ • Retrieve top 4      ║
║ • Return raw text     ║         ║ • Extract metadata    ║
╚═══════════════════════╝         ╚═══════════════════════╝
        │                                  │
        ▼                                  ▼
╔═══════════════════════╗         ╔═══════════════════════╗
║  NODE 2: chunk_text   ║         ║NODE 5: generate_answer║
╠═══════════════════════╣         ╠═══════════════════════╣
║ • Splitter: 1000 char ║         ║ • Build context       ║
║ • Overlap: 200 char   ║         ║ • Create RAG prompt   ║
║ • Add metadata        ║         ║ • Call GPT-4o         ║
║ • Track page numbers  ║         ║ • Generate citations  ║
╚═══════════════════════╝         ╚═══════════════════════╝
        │                                  │
        ▼                                  ▼
╔═══════════════════════╗         ┌───────────────────────┐
║ NODE 3: generate_     ║         │   Return Answer       │
║        embeddings     ║         │   with Citations      │
╠═══════════════════════╣         └───────────────────────┘
║ • Create ChromaDB     ║
║ • OpenAI embeddings   ║
║ • text-embedding-3-   ║
║   small model         ║
║ • Store chunks + meta ║
╚═══════════════════════╝
        │
        ▼
┌───────────────────────┐
│  Return Vector Store  │
│  Ready for Querying   │
└───────────────────────┘
```


## Key Features Explained

### Intelligent Chunking Strategy

**Problem Solved:** Long documents need to be split for processing, but naive splitting can break context.

**Solution:** RecursiveCharacterTextSplitter with:
- **1000 characters per chunk** - Optimal size for embeddings
- **200 character overlap** - Preserves context across boundaries
- **Page number tracking** - Each chunk knows its source page
- **Metadata preservation** - Filename, timestamp, and chunk index stored

### Semantic Search with ChromaDB

**How it works:**
1. User uploads PDF → Text extracted
2. Text split into chunks with metadata
3. OpenAI text-embedding-3-small creates vector embeddings
4. ChromaDB stores embeddings in in-memory collection
5. User asks question → Question embedded
6. Similarity search finds top 4 most relevant chunks
7. Chunks sent to GPT-4o for answer generation

### RAG Prompt Engineering

**The prompt structure:**

```
You are a helpful AI assistant that answers questions based on
provided context from a PDF document.

Context from PDF (filename.pdf):
[Chunk 1 from Page 3]
...text...
[Chunk 2 from Page 5]
...text...

User Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If context doesn't contain enough info, say so
- Cite sources using format: (Page X)
- Be concise but thorough
- Use markdown formatting
```

**Why this works:**
- Explicit instruction to use only provided context (prevents hallucination)
- Page number markers help with citation
- Temperature=0 ensures factual, deterministic answers
- Markdown formatting makes answers readable

### Source Citation System

**Purpose:** Enable users to verify AI-generated answers against original document.

**How it works:**
- Each chunk stores: filename, page number, chunk index
- After retrieval, citations extracted from metadata
- Citations deduplicated (same page mentioned once)
- Displayed in sidebar: "filename.pdf (Page 3)"
- Users can check retrieved chunks to verify context

**Why it matters:** Trust in AI answers requires verifiability. Citations let users fact-check the AI.

### Session-Based Architecture

**Design Choice:** No persistence, fresh start each session.

**Implications:**
- Vector store created in-memory per document
- No database needed
- Upload new document = new vector store
- Closing browser clears all data

**Benefits:**
- Simple deployment (no backend server)
- Privacy-focused (no data stored)
- Lower infrastructure costs
- Perfect for personal/demo use

## Tech Stack

* **LangGraph** - Multi-agent orchestration framework for building reliable RAG workflows
* **GPT-4o** - OpenAI's most advanced model for answer generation (temperature=0 for factual responses)
* **OpenAI Embeddings** - text-embedding-3-small model for semantic search (1536 dimensions)
* **ChromaDB** - Open-source vector database for embedding storage and similarity search
* **PyPDF** - Python library for PDF text extraction via LangChain's PyPDFLoader
* **Streamlit** - Python web framework for rapid UI development
* **Python 3.8+** - Core programming language with async support
* **python-dotenv** - Secure environment variable management



---


⭐ **Star this repo** if you find it useful!

🐛 **Issues/feedback**: https://github.com/prod-blip/aicookbook/issues
