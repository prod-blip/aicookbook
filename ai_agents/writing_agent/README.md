# AI Blogging Agent with Human-in-the-Loop

An intelligent blogging assistant that helps you create professional blog posts through a structured, multi-stage workflow with human approval at each step. Built with LangGraph and featuring web research capabilities.



https://github.com/user-attachments/assets/161c1803-fac6-4468-8709-f06a61b0b541



## Features

- **Structured Writing Workflow** - Guides you through requirements gathering, outlining, drafting, editing, and social media creation
- **Human-in-the-Loop Approval** - Pause at each stage to review, approve, or provide feedback before proceeding
- **AI-Powered Research** - Automatically enriches your content with factual information using Tavily web search
- **Multi-Format Output** - Get your blog post outline, draft, final edited version, and social media posts (LinkedIn & Twitter)
- **Interactive Interfaces** - Choose between CLI or Streamlit web interface for your preferred workflow

## Setup

### Requirements

- Python 3.9+
- OpenAI API key
- Tavily API key (for web search)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/prod-blip/aicookbook.git
cd aicookbook/ai_agents/writing_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Add your API keys to `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### API Credentials

**OpenAI API**: Get your key from [platform.openai.com](https://platform.openai.com/api-keys)

**Tavily API**: Get your key from [tavily.com](https://tavily.com) (free tier available)

## Running the Agent

### Option 1: CLI Interface

Run the command-line interface:

```bash
python blogger.py
```

**Workflow:**
1. Enter your blog topic and key pointers
2. Review the requirements brief
3. Type `continue` to proceed or provide feedback at each stage
4. Approve outline, draft, edited version, and social posts
5. Get all deliverables at the end

### Option 2: Streamlit Web Interface

Run the web interface:

```bash
streamlit run streamlit_app.py
```

**Features:**
- Visual progress tracking
- Clean, professional UI
- Download buttons for all outputs
- Session management
- Real-time API key status

## Agent Architecture

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Requirements Node  │ ◄──┐
│  (Research & Brief) │    │
└──────┬──────────────┘    │
       │                   │
       ├─(tool calls?)─────┤
       │                   │
       ▼                   │
    [Planner]          [Tools]
       │
       │ (interrupt for approval)
       ▼
┌─────────────────────┐
│   Planner Node      │ ◄──┐
│  (Create Outline)   │    │
└──────┬──────────────┘    │
       │                   │
       ├─(tool calls?)─────┤
       │                   │
       ▼                   │
    [Writer]           [Tools]
       │
       │ (interrupt for approval)
       ▼
┌─────────────────────┐
│   Writer Node       │
│  (Write Draft)      │
└──────┬──────────────┘
       │
       │ (interrupt for approval)
       ▼
┌─────────────────────┐
│   Editor Node       │
│  (Polish & Refine)  │
└──────┬──────────────┘
       │
       │ (interrupt for approval)
       ▼
┌─────────────────────┐
│   Social Node       │
│ (LinkedIn/Twitter)  │
└──────┬──────────────┘
       │
       ▼
    [END]
```

### Stage Breakdown

1. **Requirements Node**: Gathers your input, optionally uses web search to enrich context, produces content brief
2. **Planner Node**: Creates structured outline with title, intro, main sections, conclusion
3. **Writer Node**: Writes complete blog post based on approved outline
4. **Editor Node**: Polishes draft, improves clarity, fixes inconsistencies
5. **Social Node**: Generates LinkedIn post (2-3 paragraphs) and Twitter post (<280 chars)

**Key Features:**
- Memory checkpointer preserves conversation context
- Smart tool routing returns to calling node after search
- Interrupt points allow human approval before each stage
- State tracking stores all intermediate outputs


## Tech Stack

- **LangGraph** - Agent orchestration and workflow management
- **LangChain** - LLM integration and prompt management
- **OpenAI GPT-4o** - Language model for content generation
- **Tavily** - Web search API for research
- **Streamlit** - Web interface
- **Python 3.9+** - Core language

## Contributing

Found a bug or have a feature request? Open an issue on the [GitHub repository](https://github.com/prod-blip/aicookbook/issues).

## License

MIT License - see LICENSE file for details

---

**Built by Atul** | Part of the [AI Cookbook](https://github.com/prod-blip/aicookbook) collection

Give the repo a ⭐ if you find this useful!
