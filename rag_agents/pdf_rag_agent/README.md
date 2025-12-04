# PDF Query Agent - Agentic RAG System

An intelligent document Q&A assistant that processes PDF documents and answers questions using Retrieval Augmented Generation (RAG), featuring semantic search with ChromaDB and context-aware answers with source citations.

✨ **Powered by LangGraph agentic architecture with GPT-4o, OpenAI embeddings, and ChromaDB vector store**

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

### State Flow

**RAGAgentState** manages all workflow data:

```python
{
  # Inputs
  "mode": "process_document" | "query",
  "pdf_path": "/path/to/file.pdf",
  "filename": "document.pdf",
  "query": "user question",

  # Processing outputs
  "raw_text": "extracted text...",
  "page_count": 10,
  "chunks": [{text, metadata}, ...],
  "chunk_count": 25,
  "vector_store": ChromaDB_collection,

  # Query outputs
  "retrieved_chunks": [{text, metadata}, ...],
  "sources": [{filename, page, chunk_index}, ...],
  "answer": "AI-generated answer",
  "citations": ["source.pdf (Page 3)", ...],

  # Error tracking
  "errors": ["error messages"]
}
```

## Key Features Explained

### Intelligent Chunking Strategy

**Problem Solved:** Long documents need to be split for processing, but naive splitting can break context.

**Solution:** RecursiveCharacterTextSplitter with:
- **1000 characters per chunk** - Optimal size for embeddings
- **200 character overlap** - Preserves context across boundaries
- **Page number tracking** - Each chunk knows its source page
- **Metadata preservation** - Filename, timestamp, and chunk index stored

**Why this approach:**
- Prevents losing context at chunk boundaries
- Enables accurate source citations
- Balances chunk size with semantic coherence

### Semantic Search with ChromaDB

**How it works:**
1. User uploads PDF → Text extracted
2. Text split into chunks with metadata
3. OpenAI text-embedding-3-small creates vector embeddings
4. ChromaDB stores embeddings in in-memory collection
5. User asks question → Question embedded
6. Similarity search finds top 4 most relevant chunks
7. Chunks sent to GPT-4o for answer generation

**Why ChromaDB:**
- Lightweight and fast for single-document use cases
- No external dependencies or servers needed
- Built-in OpenAI embeddings integration
- In-memory mode perfect for sessions

**Why top 4 chunks:**
- Provides sufficient context (≈4000 characters)
- Stays within GPT-4o context window efficiently
- Reduces noise from less relevant chunks

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

## Important Notes

🔐 **API Key Security**: Never share your OpenAI API key. Keep it in `.env` and ensure `.env` is in `.gitignore`. The key is only used server-side.

💰 **Cost Considerations**:
- Each document processing: $0.001-$0.01 (embeddings)
- Each query: $0.01-$0.03 (GPT-4o)
- Estimated cost per document session: $0.05-$0.20
- Monitor usage at https://platform.openai.com/usage

📊 **Data Handling**:
- No data stored permanently
- Vector store lives in Streamlit session state (in-memory)
- Closing browser clears all data
- Downloaded files stored only on your machine

📄 **PDF Limitations**:
- Text-based PDFs work best
- Scanned PDFs need OCR (not included)
- Image-heavy PDFs may have limited extractable text
- Very large PDFs (>100 pages) may be slow to process

⏱️ **Processing Time**:
- PDF loading: 1-3 seconds
- Chunking: 1-2 seconds
- Embeddings generation: 5-15 seconds (depends on chunk count)
- Query processing: 3-8 seconds
- Total processing: 10-30 seconds for typical documents

🔍 **Query Quality Tips**:
- Be specific in your questions
- Reference sections or topics when known
- Ask one question at a time for best results
- Follow up questions build on previous context

## Troubleshooting

### Error: "OpenAI API key not found"

**Solutions:**
* Check that `.env` file exists in `rag_agents/pdf_rag_agent/` directory
* Verify the file contains: `OPENAI_API_KEY=sk-...`
* Ensure no extra spaces around the `=` sign
* Restart the Streamlit app after adding the key

### Error during PDF processing

**Possible causes:**
* PDF is password-protected
* PDF is corrupted or unreadable
* PDF contains only images (needs OCR)

**Solutions:**
* Try a different PDF file
* Check if PDF opens normally in a PDF viewer
* Remove password protection before uploading
* Use text-based PDFs, not scanned images

### Error: "No document has been processed yet"

**Solutions:**
* Upload and process a PDF before querying
* If you see this after processing, refresh the page and try again
* Check console for processing errors

### Answers are not accurate or relevant

**Possible causes:**
* Question too vague or broad
* Relevant information not in document
* Chunks too small to capture full context

**Solutions:**
* Be more specific in your questions
* Ask about topics you know are in the document
* Try rephrasing your question
* Check retrieved chunks to see what context was used

### ChromaDB errors or warnings

**Solutions:**
* These are usually harmless warnings
* Restart the Streamlit app
* Check Python version (3.8+ required)
* Ensure all dependencies installed correctly

### Slow performance

**Causes:**
* Large PDF with many chunks
* Network latency to OpenAI API
* Many concurrent queries

**Solutions:**
* Wait for processing to complete
* Check internet connection
* Consider smaller PDFs or specific sections
* Use local LLM for faster processing (requires code changes)

## Tech Stack

* **LangGraph** - Multi-agent orchestration framework for building reliable RAG workflows
* **GPT-4o** - OpenAI's most advanced model for answer generation (temperature=0 for factual responses)
* **OpenAI Embeddings** - text-embedding-3-small model for semantic search (1536 dimensions)
* **ChromaDB** - Open-source vector database for embedding storage and similarity search
* **PyPDF** - Python library for PDF text extraction via LangChain's PyPDFLoader
* **Streamlit** - Python web framework for rapid UI development
* **Python 3.8+** - Core programming language with async support
* **python-dotenv** - Secure environment variable management

## Future Enhancements

Planned features (contributions welcome):

- [ ] Multi-document support (query across multiple PDFs)
- [ ] Persistent vector store (save processed documents)
- [ ] OCR integration for scanned PDFs
- [ ] Adjustable chunk size and retrieval parameters
- [ ] Support for other document formats (DOCX, TXT, HTML)
- [ ] Download answers as markdown files
- [ ] Export full conversation history
- [ ] Advanced filters (by page range, sections)
- [ ] Comparison mode (compare answers across documents)
- [ ] Local LLM support (Ollama, LlamaCPP)
- [ ] Chat interface with conversational memory

---

**Built with ❤️ by Atul | Follow for more AI projects**

⭐ **Star this repo** if you find it useful!

🐛 **Issues/feedback**: https://github.com/prod-blip/aicookbook/issues
