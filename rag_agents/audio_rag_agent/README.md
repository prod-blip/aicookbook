# Audio Query Agent - Agentic RAG System

An intelligent audio Q&A assistant that transcribes audio files and answers questions using Retrieval Augmented Generation (RAG), featuring AssemblyAI transcription, semantic search with ChromaDB, and context-aware answers.


https://github.com/user-attachments/assets/817975d3-18ca-4be3-9931-c9a61b75be9c


✨ **Powered by LangGraph agentic architecture with GPT-4o, AssemblyAI, OpenAI embeddings, and ChromaDB vector store**

## Features

* **Audio Upload & Transcription** - Upload any audio file (MP3, WAV, M4A, etc.) and get it automatically transcribed via AssemblyAI
* **Semantic Search** - Find relevant information using AI-powered similarity search across your transcript chunks
* **Context-Aware Answers** - Get accurate, detailed answers based only on what was said in the audio, not generic knowledge
* **Timestamp Citations** - Every answer includes timestamps (MM:SS format) for easy verification and reference
* **Query History** - Track all your questions and answers within a session for easy reference
* **Intelligent Chunking** - Smart text splitting of transcripts with overlap ensures context preservation across chunk boundaries

## Setup

### Requirements

* Python 3.8+
* OpenAI API Key (for GPT-4o and embeddings)
* AssemblyAI API Key (for audio transcription)
* Internet connection (for API access)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/prod-blip/aicookbook.git
cd aicookbook/rag_agents/audio_rag_agent
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

   * **AssemblyAI API Key**: https://www.assemblyai.com/app/account
     - Sign up for a free account
     - Navigate to Dashboard → API Keys
     - Copy your API key
     - Free tier includes 5 hours of transcription

4. Setup your `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-proj-...your-actual-key-here
ASSEMBLYAI_API_KEY=...your-assemblyai-key-here
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

### Stage 1: Upload Audio File

1. Click "Choose an audio file" to upload your recording
2. Supported formats: MP3, WAV, M4A, MP4, OGG, FLAC, WEBM
3. Click "🎤 Transcribe & Process Audio" to start processing
4. The AI will:
   - Upload audio to AssemblyAI
   - Transcribe speech to text
   - Split transcript into 1000-character chunks with 200-character overlap
   - Generate embeddings using OpenAI text-embedding-3-small
   - Store in ChromaDB vector database with timestamp metadata
5. You'll see statistics: duration, word count, and chunk count
6. Automatically transitions to query stage

### Stage 2: Ask Questions

1. Enter your question in the text input
2. Click "🔍 Ask Question"
3. The AI will:
   - Search for the 4 most relevant transcript chunks
   - Send them to GPT-4o with your question
   - Generate a comprehensive answer
   - Include timestamp citations (MM:SS format)
4. View your answer with timestamp sources
5. Ask follow-up questions or start over with a new audio file

**What you get:**
- Natural language answer based on audio transcript
- Timestamp citations showing when information was mentioned (e.g., "at 02:15 - 03:45")
- Retrieved transcript segments for verification
- Query history for the session

**Actions available:**
- View retrieved transcript segments with timestamps
- Browse query history
- Upload new audio (resets session)

## Agent Architecture

The application uses **LangGraph** with a **single conditional graph** and **5 specialized nodes**:

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                 AUDIO QUERY AGENT WORKFLOW                  │
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
      mode="process_audio"       mode="query"
                      │                │
                      ▼                ▼

┌──────────────────────────────┐  ┌──────────────────────────────┐
│   AUDIO PROCESSING FLOW      │  │      QUERY FLOW              │
└──────────────────────────────┘  └──────────────────────────────┘

       ▼                                  ▼
╔═══════════════════════╗         ╔═══════════════════════╗
║ NODE 1: transcribe_   ║         ║ NODE 4: process_query ║
║         audio         ║         ║                       ║
╠═══════════════════════╣         ╠═══════════════════════╣
║ • Upload to           ║         ║ • Query vector store  ║
║   AssemblyAI          ║         ║ • Similarity search   ║
║ • Poll for status     ║         ║ • Retrieve top 4      ║
║ • Get transcript text ║         ║ • Extract metadata    ║
║ • Extract duration    ║         ║   (timestamps)        ║
╚═══════════════════════╝         ╚═══════════════════════╝
       │                                  │
       ▼                                  ▼
╔═══════════════════════╗         ╔═══════════════════════╗
║  NODE 2: chunk_text   ║         ║NODE 5: generate_answer║
╠═══════════════════════╣         ╠═══════════════════════╣
║ • Splitter: 1000 char ║         ║ • Build context       ║
║ • Overlap: 200 char   ║         ║ • Create RAG prompt   ║
║ • Calculate timestamp ║         ║   (audio-specific)    ║
║   ranges (MM:SS)      ║         ║ • Call GPT-4o         ║
║ • Add metadata        ║         ║ • Generate citations  ║
║                       ║         ║   with timestamps     ║
╚═══════════════════════╝         ╚═══════════════════════╝
       │                                  │
       ▼                                  ▼
╔═══════════════════════╗         ┌───────────────────────┐
║ NODE 3: generate_     ║         │   Return Answer       │
║        embeddings     ║         │   with Timestamp      │
╠═══════════════════════╣         │   Citations           │
║ • Create ChromaDB     ║         └───────────────────────┘
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

### Audio Transcription with AssemblyAI

**Problem Solved:** Audio files need to be converted to text before they can be searched and queried.

**Solution:** AssemblyAI API provides:
- **High-accuracy transcription** - Best-in-class speech-to-text model
- **Automatic punctuation & formatting** - Clean, readable transcripts
- **Multiple format support** - MP3, WAV, M4A, MP4, OGG, FLAC, WEBM
- **Async polling** - Non-blocking transcription with progress tracking
- **Cost-effective** - ~$0.015/minute (~$0.90/hour)

**How it works:**
1. Upload audio file to AssemblyAI
2. Poll every 3 seconds for completion status
3. Retrieve complete transcript with duration
4. Extract text and metadata for chunking

### Timestamp-Aware Chunking

**Problem Solved:** Audio transcripts need location references like PDFs use page numbers.

**Solution:** Linear interpolation timestamp calculation:
- **Character position mapping** - Each chunk's position in full transcript
- **Duration-based timestamps** - Calculate start/end times proportionally
- **MM:SS format** - Human-readable timestamp ranges (e.g., "02:15 - 03:45")
- **Metadata preservation** - Timestamps stored with each chunk

**Example:**
```
Total transcript: 10,000 chars, 600 seconds (10 min)
Chunk at chars 2000-3000
Start time: (2000/10000) × 600 = 120s = 02:00
End time: (3000/10000) × 600 = 180s = 03:00
Timestamp range: "02:00 - 03:00"
```

### Supported Audio Formats

**Confirmed working formats:**
- MP3 (most common)
- WAV (uncompressed, larger files)
- M4A (Apple audio format)
- MP4 (video files with audio track)
- OGG, FLAC, WEBM (modern formats)

**File size limits:**
- Streamlit default upload: 200MB
- AssemblyAI maximum: 5GB
- Recommended: Under 100MB for best performance


## Use Cases

**Perfect for:**
- Podcast Q&A and summarization
- Interview transcription and analysis
- Meeting notes and action items
- Lecture/webinar content search
- Voice memo organization
- Audio book navigation
- Research interview analysis
- Customer call insights

**Example questions:**
- "What were the main topics discussed?"
- "Summarize what was said about [specific topic]"
- "What recommendations were made?"
- "What did the speaker say about pricing?"
- "List the action items mentioned"
- "What examples were given?"

## Tech Stack

* **LangGraph** - Multi-agent orchestration framework for building reliable RAG workflows
* **GPT-4o** - OpenAI's most advanced model for answer generation (temperature=0 for factual responses)
* **AssemblyAI** - Industry-leading speech-to-text API for audio transcription
* **OpenAI Embeddings** - text-embedding-3-small model for semantic search (1536 dimensions)
* **ChromaDB** - Open-source vector database for embedding storage and similarity search
* **Streamlit** - Python web framework for rapid UI development
* **Python 3.8+** - Core programming language with async support
* **python-dotenv** - Secure environment variable management

---

⭐ **Star this repo** if you find it useful!

🐛 **Issues/feedback**: https://github.com/prod-blip/aicookbook/issues
