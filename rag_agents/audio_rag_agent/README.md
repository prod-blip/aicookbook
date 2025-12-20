# Audio Query Agent - Agentic RAG System

An intelligent audio Q&A assistant that transcribes audio files and answers questions using Retrieval Augmented Generation (RAG), featuring AssemblyAI transcription, semantic search with ChromaDB, and context-aware answers.

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

**Accuracy:** ±5-10 seconds (sufficient for most use cases)

### Intelligent Chunking Strategy

**Problem Solved:** Long transcripts need to be split for processing, but naive splitting can break context.

**Solution:** RecursiveCharacterTextSplitter with:
- **1000 characters per chunk** - Optimal size for embeddings (~150-200 words, ~45-60 seconds of speech)
- **200 character overlap** - Preserves context across boundaries
- **Timestamp range tracking** - Each chunk knows when it was spoken
- **Metadata preservation** - Filename, timestamps, and chunk index stored

### Semantic Search with ChromaDB

**How it works:**
1. User uploads audio → AssemblyAI transcribes
2. Transcript split into chunks with timestamp metadata
3. OpenAI text-embedding-3-small creates vector embeddings
4. ChromaDB stores embeddings in in-memory collection
5. User asks question → Question embedded
6. Similarity search finds top 4 most relevant chunks
7. Chunks sent to GPT-4o for answer generation with timestamp context

### RAG Prompt Engineering

**The prompt structure (audio-adapted):**

```
You are a helpful AI assistant that answers questions based on
provided context from an audio transcript.

Context from audio transcript (filename.mp3):
[Source 1, Timestamp: 02:15 - 03:45]
...transcript text...
[Source 2, Timestamp: 05:20 - 06:10]
...transcript text...

User Question: {query}

Instructions:
- Answer based ONLY on the provided transcript context
- If context doesn't contain enough info, say so
- Cite sources using format: (at MM:SS)
- Be concise but thorough
- Use markdown formatting
- Reference the audio content naturally (e.g., "The speaker mentioned at 03:45...")
```

**Why this works:**
- Explicit instruction to use only provided context (prevents hallucination)
- Timestamp markers help with citation
- Temperature=0 ensures factual, deterministic answers
- Audio-specific language ("speaker mentioned", "at timestamp")
- Markdown formatting makes answers readable

### Timestamp Citation System

**Purpose:** Enable users to verify AI-generated answers and find exact moments in audio.

**How it works:**
- Each chunk stores: filename, start_time, end_time, timestamp_range
- After retrieval, timestamp citations extracted from metadata
- Citations deduplicated (same range mentioned once)
- Displayed in sidebar: "filename.mp3 (at 02:15 - 03:45)"
- Users can check retrieved segments to verify context

**Why it matters:** Trust in AI answers requires verifiability. Timestamp citations let users fact-check by listening to the original audio at specific times.

### Session-Based Architecture

**Design Choice:** No persistence, fresh start each session.

**Implications:**
- Vector store created in-memory per audio file
- No database needed
- Upload new audio = new vector store
- Closing browser clears all data

**Benefits:**
- Simple deployment (no backend server)
- Privacy-focused (no data stored, audio files in /tmp/ only)
- Lower infrastructure costs
- Perfect for personal/demo use

## Important Notes

### Cost Considerations

**AssemblyAI Transcription:**
- Pricing: ~$0.015 per minute (~$0.90 per hour)
- Free tier: 5 hours of transcription credit
- Example costs:
  - 5-minute interview: $0.08
  - 30-minute podcast: $0.45
  - 2-hour meeting: $1.80

**OpenAI:**
- Embeddings: text-embedding-3-small (~$0.00002/1K tokens)
- GPT-4o: ~$0.0025/1K input tokens, $0.01/1K output tokens
- Typical query: $0.01-0.05 depending on context size

**Cost estimate for typical session:**
- 10-minute audio transcription: $0.15
- Embedding generation: $0.01
- 5 queries with answers: $0.10
- **Total: ~$0.26 per session**

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

### Transcription Time

**Expected processing times:**
- AssemblyAI processes at ~10-30% of audio duration
- 5-minute audio: 30-90 seconds
- 30-minute audio: 3-9 minutes
- 1-hour audio: 6-18 minutes

**Progress indicators:**
- UI shows processing steps in real-time
- Status updates during transcription polling
- Completion notification with statistics

### Limitations

**Timestamp Accuracy:**
- Uses linear interpolation (character position based)
- Accuracy: ±5-10 seconds typical
- Assumes uniform speaking pace
- Sufficient for most Q&A use cases
- For exact timestamps, consider word-level timing (future enhancement)

**Language Support:**
- Currently configured for English only
- AssemblyAI supports 100+ languages
- Easy to add language selector (see config in agent.py)

**Speaker Diarization:**
- Not enabled by default (per MVP requirements)
- Can be enabled by setting `speaker_labels=True` in TranscriptionConfig
- Would add speaker identification to citations

## Troubleshooting

### Common Issues

**1. "AssemblyAI API key not found"**
- Ensure `.env` file exists in project directory
- Check `ASSEMBLYAI_API_KEY` is set correctly
- Restart Streamlit after updating `.env`

**2. "Transcription failed: Invalid format"**
- Check file format is supported
- Try converting to MP3 using online tool or FFmpeg
- Ensure file is not corrupted

**3. "Transcription timeout"**
- Large files may take several minutes
- Check internet connection
- Verify AssemblyAI service status

**4. "No vector store available"**
- Audio must be processed before querying
- Click "Transcribe & Process Audio" first
- Check for errors during processing

**5. "OpenAI API rate limit"**
- You've hit API rate limit
- Wait a few minutes and try again
- Check your OpenAI account usage

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
