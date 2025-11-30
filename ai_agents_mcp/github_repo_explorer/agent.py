"""
GitHub Repository Explorer - LangGraph Agent with Official GitHub MCP

Uses GitHub's official MCP server (via Docker) for repository exploration.
"""

import os
import json
import asyncio
from typing import TypedDict, Annotated, List, Dict, Any
import operator

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


# ============= STATE DEFINITION =============

class GitHubExplorerState(TypedDict):
    """State for GitHub Repository Explorer Agent"""
    # Inputs
    repo_url: str
    github_token: str
    user_query: str

    # MCP Connection
    mcp_session: Any  # MCP ClientSession
    stdio_context: Any  # stdio context manager for cleanup
    available_tools: List[Dict]  # Tools from GitHub MCP server

    # Results
    tool_results: Annotated[List[Dict], operator.add]
    explanation: str

    # Error tracking
    errors: Annotated[List[str], operator.add]

    # LLM messages
    messages: Annotated[list, operator.add]


# ============= AGENT NODES =============
# Note: MCP initialization is now handled in run_agent() using async with


async def query_github_mcp(state: GitHubExplorerState) -> GitHubExplorerState:
    """
    Use LLM to understand query and call appropriate GitHub MCP tools.

    Reads: state["user_query"], state["repo_url"], state["mcp_session"], state["available_tools"]
    Updates: state["tool_results"], state["messages"]
    """
    print("\n=== Querying GitHub MCP ===")

    if not state.get("mcp_session"):
        print("⚠️ Skipping - MCP not connected")
        return {**state, "tool_results": []}

    try:
        session = state["mcp_session"]
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )

        # Create tool descriptions for LLM
        tools_desc = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in state["available_tools"][:10]  # Limit to avoid token overflow
        ])

        # Extract owner/repo from URL
        repo_parts = state['repo_url'].rstrip('/').split('/')
        owner = repo_parts[-2] if len(repo_parts) >= 2 else ""
        repo = repo_parts[-1] if len(repo_parts) >= 1 else ""

        prompt = f"""You are helping explore a GitHub repository using the official GitHub MCP server.

Repository: {state['repo_url']}
Owner: {owner}
Repo: {repo}
User Query: {state['user_query']}

Available GitHub MCP Tools (USE THESE EXACT NAMES):
{tools_desc}

CRITICAL: You MUST use the EXACT tool names listed above. Do NOT invent tool names.

Common queries and recommended tools:
- "What is this repository about?" → use search_repositories with query="{owner}/{repo}"
- "Show me issues" → use list_issues with owner="{owner}", repo="{repo}", state="OPEN"
- "Show me all issues" → use list_issues with owner="{owner}", repo="{repo}", state="OPEN" (list both open and closed separately if needed)
- "Show me pull requests" → use list_pull_requests with owner="{owner}", repo="{repo}", state="OPEN"
- "Show me the code" → use get_file_contents or search_code
- "What are recent commits?" → use list_commits with owner="{owner}", repo="{repo}"

IMPORTANT Parameter Rules:
- For list_issues and list_pull_requests: state must be "OPEN" or "CLOSED" (uppercase), NOT "all" or "open"
- For owner and repo: extract from the repository URL
- Always use exact parameter names from the tool schema

Based on the user's query, determine which MCP tool(s) to call and with what parameters.

Return ONLY a JSON array of tool calls (no markdown, no explanation).

Example format:
[
    {{"tool": "search_repositories", "params": {{"query": "{owner}/{repo}"}}}},
    {{"tool": "list_issues", "params": {{"owner": "{owner}", "repo": "{repo}", "state": "OPEN"}}}}
]

Remember: Use "OPEN" or "CLOSED" (uppercase) for state parameter!

JSON array:"""

        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        # Clean markdown if present
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        # Parse tool calls
        tool_calls = json.loads(content)
        print(f"📋 LLM planned {len(tool_calls)} tool calls")

        # Execute each tool call via MCP
        results = []
        for tool_call in tool_calls[:3]:  # Limit to 3 calls
            tool_name = tool_call["tool"]
            params = tool_call.get("params", {})

            print(f"🔄 Calling {tool_name}...")

            try:
                result = await session.call_tool(tool_name, params)

                # Convert MCP result content to serializable format
                result_content = []
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        result_content.append({"type": "text", "text": content_item.text})
                    elif hasattr(content_item, 'model_dump'):
                        result_content.append(content_item.model_dump())
                    else:
                        result_content.append(str(content_item))

                results.append({
                    "tool": tool_name,
                    "params": params,
                    "result": result_content,
                    "success": True
                })

                print(f"✅ {tool_name} completed")

            except Exception as e:
                print(f"❌ {tool_name} failed: {e}")
                results.append({
                    "tool": tool_name,
                    "params": params,
                    "error": str(e),
                    "success": False
                })

        return {
            **state,
            "tool_results": results,
            "messages": [response]
        }

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return {
            **state,
            "errors": [f"Query planning failed: Invalid JSON from LLM"]
        }

    except Exception as e:
        print(f"❌ Query error: {e}")
        return {
            **state,
            "errors": [f"Query execution failed: {str(e)}"]
        }


async def generate_explanation(state: GitHubExplorerState) -> GitHubExplorerState:
    """
    Generate natural language explanation from MCP results.

    Reads: state["user_query"], state["tool_results"]
    Updates: state["explanation"], state["messages"]
    """
    print("\n=== Generating Explanation ===")

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7
        )

        # Prepare results summary (limit to avoid token overflow)
        results_text = []
        for result in state["tool_results"]:
            if result.get("success"):
                # Extract text from result content items
                result_content = result.get('result', [])
                text_parts = []

                for item in result_content[:3]:  # Limit to first 3 items
                    if isinstance(item, dict) and 'text' in item:
                        text = item['text']
                        # Limit each text to 500 characters
                        if len(text) > 500:
                            text = text[:500] + "... (truncated)"
                        text_parts.append(text)
                    elif isinstance(item, str):
                        text = item[:500]
                        text_parts.append(text)

                results_text.append(f"""
Tool: {result['tool']}
Result: {' '.join(text_parts)}
""")

        results_summary = "\n---\n".join(results_text) if results_text else "No results available"

        # Further limit results_summary if still too long
        if len(results_summary) > 2000:
            results_summary = results_summary[:2000] + "\n\n... (results truncated due to length)"

        prompt = f"""Answer the user's question about this GitHub repository based on the MCP tool results.

Repository: {state['repo_url']}
User Query: {state['user_query']}

MCP Tool Results:
{results_summary}

Provide a clear, helpful explanation that:
1. **Directly answers** the user's question
2. **Uses data** from the MCP results
3. **Provides context** about the repository
4. **Suggests** related exploration if relevant

Use markdown formatting. Be concise but thorough.
"""

        response = await llm.ainvoke(prompt)

        print("✅ Explanation generated")

        return {
            **state,
            "explanation": response.content,
            "messages": [response]
        }

    except Exception as e:
        print(f"❌ Explanation generation error: {e}")
        return {
            **state,
            "explanation": f"Error generating explanation: {str(e)}",
            "errors": [f"Explanation failed: {str(e)}"]
        }


# Note: Cleanup is handled automatically when async with exits


# ============= MAIN RUNNER =============

async def run_agent(repo_url: str, github_token: str, user_query: str) -> Dict:
    """
    Main function to run the GitHub repository explorer agent.

    Args:
        repo_url: GitHub repository URL
        github_token: GitHub personal access token
        user_query: User's natural language query

    Returns:
        Dictionary with exploration results
    """
    print("\n" + "=" * 60)
    print("🔍 GITHUB REPOSITORY EXPLORER (Official MCP)")
    print("=" * 60)

    # Setup MCP server parameters
    server_params = StdioServerParameters(
        command="docker",
        args=[
            "run", "-i", "--rm",
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server"
        ],
        env={
            "GITHUB_PERSONAL_ACCESS_TOKEN": github_token or os.getenv("GITHUB_TOKEN", "")
        }
    )

    try:
        print("🚀 Starting GitHub MCP server (Docker)...")

        # Use async with to keep background tasks alive
        async with stdio_client(server_params) as (read_stream, write_stream):
            print("✅ MCP server started")

            # Wait for container to fully start
            await asyncio.sleep(2)

            # Create session
            print("📡 Creating MCP session...")
            session = ClientSession(read_stream, write_stream)

            # IMPORTANT: Session must also be used as async context manager
            async with session:
                print("✅ Session context started")

                # Initialize session
                print("🔌 Initializing session...")
                await session.initialize()
                print("✅ Session initialized")

                # Get available tools
                print("📋 Listing tools...")
                tools_response = await session.list_tools()
                available_tools = [
                    {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                    for t in tools_response.tools
                ]
                print(f"✅ Found {len(available_tools)} tools")

                # Now run the agent workflow with MCP session
                initial_state = {
                    "repo_url": repo_url,
                    "github_token": github_token or os.getenv("GITHUB_TOKEN", ""),
                    "user_query": user_query,
                    "mcp_session": session,
                    "stdio_context": None,  # Not needed anymore
                    "available_tools": available_tools,
                    "tool_results": [],
                    "explanation": "",
                    "errors": [],
                    "messages": []
                }

                # Create simplified graph (without MCP initialization/cleanup nodes)
                workflow = StateGraph(GitHubExplorerState)
                workflow.add_node("query_mcp", query_github_mcp)
                workflow.add_node("generate_explanation", generate_explanation)
                workflow.set_entry_point("query_mcp")
                workflow.add_edge("query_mcp", "generate_explanation")
                workflow.add_edge("generate_explanation", END)
                graph = workflow.compile()

                # Run the graph
                result = await graph.ainvoke(initial_state)

                print("✅ Agent workflow completed")

                return {
                    "repo_url": repo_url,
                    "user_query": user_query,
                    "explanation": result["explanation"],
                    "tool_results": result["tool_results"],
                    "available_tools": available_tools,
                    "errors": result["errors"],
                    "success": len(result["errors"]) == 0
                }

    except Exception as e:
        print(f"\n❌ Agent execution error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "repo_url": repo_url,
            "user_query": user_query,
            "explanation": f"Error: {str(e)}",
            "tool_results": [],
            "available_tools": [],
            "errors": [str(e)],
            "success": False
        }


# ============= TESTING =============

if __name__ == "__main__":
    # Test the agent
    async def test():
        result = await run_agent(
            repo_url="https://github.com/facebook/react",
            github_token=os.getenv("GITHUB_TOKEN", ""),
            user_query="Show me recent issues"
        )

        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(result["explanation"])

        if result["errors"]:
            print("\nErrors:")
            for error in result["errors"]:
                print(f"  - {error}")

    asyncio.run(test())
