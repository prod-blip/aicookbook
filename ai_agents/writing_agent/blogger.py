import uuid
from typing import Annotated, Literal, Optional, Dict
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# ✅ Tavily (LangChain official)
from langchain_community.tools.tavily_search import TavilySearchResults
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace
import os

tracer_provider = register(
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
    api_key=os.getenv("PHOENIX_API_KEY"),
    project_name="blogger-agent", # name this to whatever you would like
    auto_instrument=True,
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = trace.get_tracer(__name__)

# -------------------------
# Tools
# -------------------------

tavily_search = TavilySearchResults(
    max_results=3,
    description="Search the web for factual information and context"
)

# -------------------------
# State
# -------------------------

class BlogState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    raw_pointers: Optional[str]
    enriched_pointers: Optional[str]
    outline: Optional[str]
    blog_draft: Optional[str]
    edited_blog: Optional[str]
    social_posts: Optional[Dict[str, str]]

    approved_stage: Optional[
        Literal["requirements", "outline", "draft", "edit"]
    ]

    # Track which node called tools
    calling_node: Optional[str]


# -------------------------
# LLM
# -------------------------

llm = ChatOpenAI(temperature=0)

# -------------------------
# Node 1 — Requirements + Research
# -------------------------

def requirements_node(state: BlogState) -> BlogState:
    """
    Collate user's raw blog pointers and enrich with research.

    Reads: state["messages"]
    Updates: state["messages"], state["raw_pointers"]
    """
    # Only show print if we're not coming back from tools
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    # Skip if last message was a tool response (we already processed)
    if last_msg and last_msg.type == "tool":
        print("🔄 Processing tool results...")
    else:
        print("\n=== 📋 Requirements Gathering ===")

    requirements_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a research assistant.

Tasks:
1. Collate and organize the user's raw blog pointers
2. Use tavily_search ONLY if you need additional factual information
3. Produce a refined, factual content brief

Format your response clearly with sections and bullet points.
Once you have gathered requirements, do NOT call tools again.
"""
            ),
            ("placeholder", "{messages}")
        ]
    )

    chain = requirements_prompt | llm.bind_tools([tavily_search])
    response = chain.invoke(state)

    # Only print completion if not making tool calls
    if not (hasattr(response, 'tool_calls') and response.tool_calls):
        print("✅ Requirements gathered")
        calling_node = None
    else:
        calling_node = "requirements"

    return {
        **state,
        "messages": [response],
        "calling_node": calling_node
    }

# -------------------------
# Node 2 — Blog Planner
# -------------------------

def planner_node(state: BlogState) -> BlogState:
    """
    Create a blog post outline based on requirements.

    Reads: state["messages"]
    Updates: state["messages"], state["outline"]
    """
    # Check if coming back from tools
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    if last_msg and last_msg.type == "tool":
        print("🔄 Processing tool results...")
    else:
        print("\n=== 📝 Creating Outline ===")

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a technical content strategist.

Create a blog post outline that includes:
- Title
- Introduction
- Main sections (3-5 sections)
- Conclusion

The outline should be clear and logical.
Use tavily_search ONLY if you need critical additional context.

Do NOT write the full blog, just the outline.
Once you have the outline, do NOT call tools again.
"""
            ),
            ("placeholder", "{messages}")
        ]
    )

    chain = planner_prompt | llm.bind_tools([tavily_search])
    response = chain.invoke(state)

    # Only print completion and save outline if not making tool calls
    if not (hasattr(response, 'tool_calls') and response.tool_calls):
        print("✅ Outline created")
        outline_content = response.content if hasattr(response, 'content') else str(response)
        calling_node = None
    else:
        outline_content = state.get("outline")  # Keep existing outline
        calling_node = "planner"

    return {
        **state,
        "messages": [response],
        "outline": outline_content,
        "calling_node": calling_node
    }

# -------------------------
# Node 3 — Blog Writer
# -------------------------

def writer_node(state: BlogState) -> BlogState:
    """
    Write a detailed blog post based on the outline.

    Reads: state["messages"], state["outline"]
    Updates: state["messages"], state["blog_draft"]
    """
    print("\n=== ✍️ Writing Blog Draft ===")

    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert technical writer.

Using the approved outline, write a detailed blog post
for a sophisticated audience.

Focus on:
- Depth and technical accuracy
- Clear explanations
- Engaging storytelling
- Practical examples

Write the complete blog post with all sections.
"""
            ),
            ("placeholder", "{messages}")
        ]
    )

    chain = writer_prompt | llm
    response = chain.invoke(state)

    print("✅ Blog draft completed")

    return {
        **state,
        "messages": [response],
        "blog_draft": response.content if hasattr(response, 'content') else str(response)
    }

# -------------------------
# Node 4 — Blog Editor
# -------------------------

def editor_node(state: BlogState) -> BlogState:
    """
    Edit and refine the blog draft based on feedback.

    Reads: state["messages"], state["blog_draft"]
    Updates: state["messages"], state["edited_blog"]
    """
    print("\n=== 🔍 Editing Blog ===")

    editor_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a professional editor.

Review and improve the blog post:
- Improve clarity and flow
- Fix grammar and inconsistencies
- Enhance readability
- Preserve the author's voice and intent

If the user provided specific feedback, incorporate it.
Otherwise, do a general polish pass.
"""
            ),
            ("placeholder", "{messages}")
        ]
    )

    chain = editor_prompt | llm
    response = chain.invoke(state)

    print("✅ Blog edited")

    return {
        **state,
        "messages": [response],
        "edited_blog": response.content if hasattr(response, 'content') else str(response)
    }

# -------------------------
# Node 5 — Social Media Generator
# -------------------------

def social_node(state: BlogState) -> BlogState:
    """
    Generate social media posts for the blog.

    Reads: state["messages"], state["edited_blog"]
    Updates: state["messages"], state["social_posts"]
    """
    print("\n=== 📱 Generating Social Media Posts ===")

    social_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a social media marketing expert.

Generate promotional posts for the blog:
- 1 LinkedIn post (professional, thoughtful, 2-3 paragraphs)
- 1 X/Twitter post (concise, engaging, under 280 characters)

Format your response clearly with headers:
## LinkedIn Post
[content]

## Twitter Post
[content]

Goal: drive traffic to the blog with compelling hooks.
"""
            ),
            ("placeholder", "{messages}")
        ]
    )

    chain = social_prompt | llm
    response = chain.invoke(state)

    print("✅ Social media posts generated")

    return {
        **state,
        "messages": [response],
        "social_posts": {
            "content": response.content if hasattr(response, 'content') else str(response)
        }
    }

# -------------------------
# Graph
# -------------------------

def route_after_requirements(state: BlogState) -> Literal["tools", "planner"]:
    """Route to tools if needed, otherwise to planner."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "tools"
    return "planner"

def route_after_planner(state: BlogState) -> Literal["tools", "writer"]:
    """Route to tools if needed, otherwise to writer."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "tools"
    return "writer"

def route_after_tools(state: BlogState) -> Literal["requirements", "planner"]:
    """Route back to the node that called the tools."""
    calling_node = state.get("calling_node")
    if calling_node == "planner":
        return "planner"
    return "requirements"  # Default to requirements

builder = StateGraph(BlogState)

# Add nodes
builder.add_node("requirements", requirements_node)
builder.add_node("planner", planner_node)
builder.add_node("writer", writer_node)
builder.add_node("editor", editor_node)
builder.add_node("social", social_node)
builder.add_node("tools", ToolNode([tavily_search]))

# Build flow
builder.add_edge(START, "requirements")

# Requirements can use tools or proceed to planner
builder.add_conditional_edges("requirements", route_after_requirements, {
    "tools": "tools",
    "planner": "planner"
})

# Planner can use tools or proceed to writer
builder.add_conditional_edges("planner", route_after_planner, {
    "tools": "tools",
    "writer": "writer"
})

# After tools, route back to the calling node
builder.add_conditional_edges("tools", route_after_tools, {
    "requirements": "requirements",
    "planner": "planner"
})

# Linear flow after writer
builder.add_edge("writer", "editor")
builder.add_edge("editor", "social")
builder.add_edge("social", END)

# Compile with memory checkpointer for human-in-the-loop
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["planner", "writer", "editor", "social"]
)

# -------------------------
# Runner
# -------------------------

def run():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n📝 Blogging Agent with Human-in-the-Loop")
    print("=" * 60)
    print("This agent will pause at each stage for your approval.")
    print("Type 'continue' to proceed, 'q' to quit, or provide feedback.\n")

    # Initial input
    user_input = input("📋 Enter your blog topic/pointers: ")
    if user_input.lower() == "q":
        return

    # Stream through requirements
    for event in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values"
    ):
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, 'content') and last_msg.type == "ai":
                print(f"\n🤖 Assistant:\n{last_msg.content}\n")

    # Main loop for human-in-the-loop approval
    while True:
        try:
            # Get current state
            snapshot = graph.get_state(config)
            next_node = snapshot.next[0] if snapshot.next else None

            if not next_node:
                print("\n✅ Blog creation complete!")
                print("\nFinal outputs available in state:")
                if snapshot.values.get("outline"):
                    print("  - Outline")
                if snapshot.values.get("blog_draft"):
                    print("  - Blog Draft")
                if snapshot.values.get("edited_blog"):
                    print("  - Edited Blog")
                if snapshot.values.get("social_posts"):
                    print("  - Social Media Posts")
                break

            # Show what's next
            stage_names = {
                "planner": "Creating Outline",
                "writer": "Writing Blog Draft",
                "editor": "Editing Blog",
                "social": "Generating Social Posts"
            }

            print(f"\n🔄 Next stage: {stage_names.get(next_node, next_node)}")
            user_input = input("Type 'continue' to proceed, or provide feedback: ")

            if user_input.lower() == "q":
                break

            # Stream the next stage
            if user_input.lower() == "continue" or user_input.strip() == "":
                for event in graph.stream(None, config, stream_mode="values"):
                    if "messages" in event and event["messages"]:
                        last_msg = event["messages"][-1]
                        if hasattr(last_msg, 'content') and last_msg.type == "ai":
                            print(f"\n🤖 Assistant:\n{last_msg.content}\n")
            else:
                # User provided feedback
                for event in graph.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config,
                    stream_mode="values"
                ):
                    if "messages" in event and event["messages"]:
                        last_msg = event["messages"][-1]
                        if hasattr(last_msg, 'content') and last_msg.type == "ai":
                            print(f"\n🤖 Assistant:\n{last_msg.content}\n")

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

    print("\n" + "=" * 60)
    print("Session ended. Thread ID:", thread_id)


if __name__ == "__main__":
    run()
