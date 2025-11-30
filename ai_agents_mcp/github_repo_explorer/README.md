# GitHub Repository Explorer with Official MCP

An AI-powered GitHub repository explorer that uses **GitHub's Official Model Context Protocol (MCP) server** with LangGraph for intelligent query planning. Ask natural language questions about any repository and get AI-powered insights.

✨ **Powered by GitHub's Official MCP Server, LangGraph, and GPT-4o**




https://github.com/user-attachments/assets/ea724cf9-619c-4689-803a-43381b46df3a





## Features

* **Official GitHub MCP Server** - Uses GitHub's production-ready MCP server (via Docker) for real-time repository access
* **LangGraph Orchestration** - Intelligent agent workflow with query planning, tool execution, and result synthesis
* **Natural Language Queries** - Ask questions in plain English - "What issues are labeled as bugs?", "Show me recent merged PRs"
* **Real-Time GitHub API** - Direct access to GitHub's API through official MCP tools (issues, PRs, repos, and more)
* **AI-Powered Analysis** - GPT-4o understands your query, selects appropriate tools, and provides clear explanations
* **Docker-Based** - MCP server runs in Docker container for isolation and consistency
* **Streamlit Interface** - Clean, intuitive UI following AI Cookbook patterns

## Setup

### Requirements

* **Docker** - Must be installed and running
* **Python 3.8+**
* **OpenAI API Key** - For GPT-4o agent
* **GitHub Personal Access Token** - For GitHub API access via MCP

### Installation

1. **Install Docker**

   - macOS: Download from [docker.com](https://www.docker.com/products/docker-desktop)
   - Linux: `sudo apt-get install docker.io`
   - Windows: Download Docker Desktop

   Verify installation:
   ```bash
   docker --version
   ```

2. **Clone this repository**

   ```bash
   git clone https://github.com/prod-blip/aicookbook.git
   cd aicookbook/ai_agents_mcp/github_repo_explorer
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Get your API credentials**

   * **OpenAI API Key**: https://platform.openai.com/api-keys
   * **GitHub Token**: https://github.com/settings/tokens
     - Click "Generate new token (classic)"
     - Select scopes: `repo` (for full repository access) or `public_repo` (for public only)
     - Copy the token immediately (you won't see it again!)

5. **Setup `.env` file**

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:
   ```env
   OPENAI_API_KEY=sk-proj-...your-openai-key
   GITHUB_TOKEN=ghp_...your-github-token
   ```

## Running the App

1. **Start Docker** (if not already running)

2. **Run Streamlit app**

   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open browser** to `http://localhost:8501`

4. **Enter repository URL and ask questions!**

   Example:
   - Repository: `https://github.com/facebook/react`
   - Query: "What are the most discussed issues?"

## How It Works

### Architecture Overview

```
User Query → Streamlit UI → LangGraph Agent → GitHub MCP Server → GitHub API
                                                  (Docker)
```

### LangGraph Workflow (2 Nodes)

```
┌────────────────────────────────────────────────┐
│ MCP Server Initialization (async with)        │
│    • Starts GitHub MCP server in Docker       │
│    • Connects via stdio protocol              │
│    • Lists 40 available MCP tools             │
└──────────────┬─────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────┐
│ 1. query_github_mcp                            │
│    • GPT-4o analyzes user query               │
│    • Plans which MCP tools to call            │
│    • Executes tools via MCP session           │
│    • Collects and formats results             │
└──────────────┬─────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────┐
│ 2. generate_explanation                        │
│    • GPT-4o synthesizes MCP results           │
│    • Creates natural language answer          │
│    • Provides context and insights            │
└──────────────┬─────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────┐
│ Automatic Cleanup (async with exits)          │
│    • Closes MCP session                        │
│    • Terminates Docker container              │
└────────────────────────────────────────────────┘
```

### Available GitHub MCP Tools (40 Total)

The official GitHub MCP server provides these tools:

**Repository Management:**
- `search_repositories` - Find repositories by keyword
- `create_repository` - Create new repositories
- `fork_repository` - Fork existing repositories
- `get_me` - Get authenticated user information

**Issues:**
- `list_issues` - List issues (state: OPEN/CLOSED)
- `search_issues` - Search issues with filters
- `add_issue_comment` - Add comments to issues
- `issue_read` - Read issue details
- `issue_write` - Create/update issues
- `list_issue_types` - List available issue types
- `sub_issue_write` - Manage sub-issues
- `assign_copilot_to_issue` - Assign GitHub Copilot

**Pull Requests:**
- `list_pull_requests` - List PRs (state: OPEN/CLOSED)
- `search_pull_requests` - Search PRs with filters
- `create_pull_request` - Create new PRs
- `update_pull_request` - Update existing PRs
- `update_pull_request_branch` - Update PR branches
- `merge_pull_request` - Merge PRs
- `pull_request_read` - Read PR details
- `pull_request_review_write` - Add PR reviews
- `add_comment_to_pending_review` - Comment on reviews
- `request_copilot_review` - Request Copilot review

**Code & Files:**
- `get_file_contents` - Read file contents
- `create_or_update_file` - Modify files
- `delete_file` - Delete files
- `push_files` - Push multiple files
- `search_code` - Search code in repositories

**Branches & Commits:**
- `list_branches` - List repository branches
- `create_branch` - Create new branches
- `list_commits` - List commit history
- `get_commit` - Get specific commit details

**Releases & Tags:**
- `list_releases` - List all releases
- `get_latest_release` - Get latest release
- `get_release_by_tag` - Get specific release
- `list_tags` - List repository tags
- `get_tag` - Get specific tag

**Labels & Teams:**
- `get_label` - Get label information
- `get_teams` - List organization teams
- `get_team_members` - List team members

**Users:**
- `search_users` - Search GitHub users

## Example Queries (Based on Actual MCP Tools)

### Repository Analysis
```
✅ "What is this repository about?"
   → Uses: search_repositories

✅ "Show me repository information"
   → Uses: search_repositories

✅ "Find repositories related to React"
   → Uses: search_repositories
```

### Issues Management
```
✅ "Show me open issues"
   → Uses: list_issues (state="OPEN")

✅ "List all recent issues"
   → Uses: list_issues (state="OPEN")

✅ "Search for bug-related issues"
   → Uses: search_issues

✅ "What issues need attention?"
   → Uses: list_issues, issue_read
```

### Pull Requests
```
✅ "Show me open pull requests"
   → Uses: list_pull_requests (state="OPEN")

✅ "List merged PRs"
   → Uses: list_pull_requests (state="CLOSED")

✅ "Find PRs that need review"
   → Uses: search_pull_requests

✅ "Show recent PR activity"
   → Uses: list_pull_requests
```

### Code Exploration
```
✅ "Show me the README file"
   → Uses: get_file_contents

✅ "Find authentication code"
   → Uses: search_code

✅ "Search for API endpoints"
   → Uses: search_code

✅ "Get contents of package.json"
   → Uses: get_file_contents
```

### Commits & Branches
```
✅ "Show me recent commits"
   → Uses: list_commits

✅ "List all branches"
   → Uses: list_branches

✅ "Get commit details"
   → Uses: get_commit

✅ "What's the latest commit?"
   → Uses: list_commits
```

### Releases & Tags
```
✅ "Show me the latest release"
   → Uses: get_latest_release

✅ "List all releases"
   → Uses: list_releases

✅ "Get version tags"
   → Uses: list_tags

✅ "Show release notes"
   → Uses: get_release_by_tag
```

### Team & User Information
```
✅ "Who are the contributors?"
   → Uses: search_users

✅ "List team members"
   → Uses: get_team_members

✅ "Show organization teams"
   → Uses: get_teams
```

## Important Notes

🐳 **Docker Requirement**:
- GitHub MCP server runs in Docker container
- Docker must be running before starting the app
- First run downloads the Docker image (~100MB)
- Container is removed after each query (`--rm` flag)

🔐 **API Key Security**:
- Never commit `.env` file to version control
- Keys are passed securely to Docker via environment variables
- GitHub token scope determines what repositories you can access

💰 **Cost Considerations**:
- Each query uses 2 GPT-4o API calls (planning + synthesis)
- Estimated cost: $0.02-$0.03 per query
- Monitor usage at https://platform.openai.com/usage
- GitHub API calls are free (within rate limits)

⏱️ **Processing Time**:
- Docker startup: 2-5 seconds (first time: ~30 seconds for image pull)
- MCP tool execution: 3-10 seconds
- AI analysis: 5-10 seconds
- Total: 10-25 seconds per query

🔄 **Rate Limits**:
- With GitHub token: 5000 requests/hour
- Without token: 60 requests/hour
- Each query typically uses 1-3 GitHub API calls


## Tech Stack

* **GitHub Official MCP Server** - Production MCP server from GitHub (Docker)
* **LangGraph** - Multi-agent orchestration framework
* **GPT-4o** - OpenAI's latest model for planning and synthesis
* **MCP Python SDK** - Official Model Context Protocol client
* **Streamlit** - Python web framework for UI
* **Docker** - Container runtime for MCP server
* **python-dotenv** - Environment variable management

## Comparison with Custom MCP

### This Implementation (Official MCP)
✅ Uses GitHub's battle-tested MCP server
✅ Automatically updated by GitHub
✅ Full feature parity with GitHub API
✅ Production-ready and reliable
✅ No maintenance burden

### Custom MCP Alternative
⚠️ Need to implement all tools yourself
⚠️ Must keep up with GitHub API changes
⚠️ More code to maintain
⚠️ Potential bugs and edge cases

**Recommendation**: Use official MCP when available!

## Future Enhancements

- [ ] Support for GitHub Organizations
- [ ] Repository comparison feature
- [ ] Scheduled monitoring and alerts
- [ ] Export analysis to PDF
- [ ] Save query history
- [ ] Multi-repository batch queries
- [ ] Custom MCP tool chaining
- [ ] Webhook integration for real-time updates

## License

This project is part of the AI Cookbook repository. See the main repository for license information.

---


⭐ **If you find this useful, please star the [AI Cookbook repository](https://github.com/prod-blip/aicookbook)!**
