import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import uuid

# Load environment variables
load_dotenv()

# Import the graph from blogger.py
from blogger import graph

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="AI Blogging Agent",
    page_icon="📝",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stage-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key Status
    st.subheader("API Keys Status")

    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if openai_key:
        st.success("✅ OpenAI API Key")
    else:
        st.error("❌ OpenAI API Key Missing")

    if tavily_key:
        st.success("✅ Tavily API Key")
    else:
        st.error("❌ Tavily API Key Missing")

    st.markdown("---")

    # What This Does
    with st.expander("ℹ️ What This Does"):
        st.markdown("""
        This AI Blogging Agent helps you create professional blog posts through a structured workflow:

        **Stages:**
        1. 📋 Requirements - Gather and research your topic
        2. 📝 Outline - Create a structured outline
        3. ✍️ Draft - Write the full blog post
        4. 🔍 Edit - Polish and refine the content
        5. 📱 Social - Generate promotional posts

        You can approve or provide feedback at each stage!
        """)

    st.markdown("---")

    # Reset Session
    if st.button("🔄 Start New Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.caption("**Tech Stack:** LangGraph • OpenAI • Tavily • Streamlit")

# -------------------------
# Session State Initialization
# -------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "current_stage" not in st.session_state:
    st.session_state.current_stage = "input"

if "requirements_done" not in st.session_state:
    st.session_state.requirements_done = False

if "outline" not in st.session_state:
    st.session_state.outline = None

if "blog_draft" not in st.session_state:
    st.session_state.blog_draft = None

if "edited_blog" not in st.session_state:
    st.session_state.edited_blog = None

if "social_posts" not in st.session_state:
    st.session_state.social_posts = None

if "last_response" not in st.session_state:
    st.session_state.last_response = None

# -------------------------
# Main App
# -------------------------

st.markdown('<p class="main-header">📝 AI Blogging Agent</p>', unsafe_allow_html=True)
st.markdown("### Create professional blog posts with AI assistance")

# Check API keys
if not openai_key or not tavily_key:
    st.error("⚠️ Please set up your API keys in the `.env` file to continue.")
    st.info("Required: `OPENAI_API_KEY` and `TAVILY_API_KEY`")
    st.stop()

# Progress indicator
stage_mapping = {
    "input": 0,
    "requirements": 1,
    "planner": 2,
    "writer": 3,
    "editor": 4,
    "social": 5,
    "complete": 6
}

progress_value = stage_mapping.get(st.session_state.current_stage, 0) / 6
st.progress(progress_value)

# Config for graph
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# -------------------------
# Stage 1: Initial Input
# -------------------------

if st.session_state.current_stage == "input":
    st.markdown("### 📋 Step 1: Enter Your Blog Topic")

    topic_input = st.text_area(
        "Describe your blog topic or provide key pointers:",
        placeholder="Example: Write a blog about the benefits of using LangGraph for building AI agents...",
        height=150
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Blog Creation", type="primary"):
            if topic_input.strip():
                with st.spinner("🔄 Processing requirements..."):
                    # Stream through requirements stage
                    try:
                        for event in graph.stream(
                            {"messages": [HumanMessage(content=topic_input)]},
                            config,
                            stream_mode="values"
                        ):
                            if "messages" in event and event["messages"]:
                                last_msg = event["messages"][-1]
                                if hasattr(last_msg, 'content') and last_msg.type == "ai":
                                    st.session_state.last_response = last_msg.content

                        st.session_state.requirements_done = True
                        st.session_state.current_stage = "planner"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a blog topic first.")

# -------------------------
# Stage 2-5: Approval Stages
# -------------------------

elif st.session_state.current_stage in ["planner", "writer", "editor", "social"]:

    # Get current snapshot
    snapshot = graph.get_state(config)

    # Stage information
    stage_info = {
        "planner": {
            "title": "📝 Step 2: Review Outline",
            "description": "Review the blog post outline created by the AI.",
            "emoji": "📝"
        },
        "writer": {
            "title": "✍️ Step 3: Review Draft",
            "description": "Review the complete blog post draft.",
            "emoji": "✍️"
        },
        "editor": {
            "title": "🔍 Step 4: Review Edited Version",
            "description": "Review the polished and edited blog post.",
            "emoji": "🔍"
        },
        "social": {
            "title": "📱 Step 5: Review Social Media Posts",
            "description": "Review the promotional social media content.",
            "emoji": "📱"
        }
    }

    current_info = stage_info[st.session_state.current_stage]

    st.markdown(f"### {current_info['title']}")
    st.markdown(current_info['description'])

    # Display last response
    if st.session_state.last_response:
        with st.container():
            st.markdown('<div class="stage-box">', unsafe_allow_html=True)
            st.markdown(f"**{current_info['emoji']} AI Response:**")
            st.markdown(st.session_state.last_response)
            st.markdown('</div>', unsafe_allow_html=True)

    # Feedback section
    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col1:
        feedback = st.text_area(
            "💬 Provide feedback (optional):",
            placeholder="Enter any changes or improvements you'd like...",
            height=100,
            key=f"feedback_{st.session_state.current_stage}"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✅ Approve & Continue", type="primary", key=f"approve_{st.session_state.current_stage}"):
            with st.spinner(f"🔄 Processing {st.session_state.current_stage}..."):
                try:
                    # Stream the next stage
                    if feedback.strip():
                        # User provided feedback
                        for event in graph.stream(
                            {"messages": [HumanMessage(content=feedback)]},
                            config,
                            stream_mode="values"
                        ):
                            if "messages" in event and event["messages"]:
                                last_msg = event["messages"][-1]
                                if hasattr(last_msg, 'content') and last_msg.type == "ai":
                                    st.session_state.last_response = last_msg.content
                    else:
                        # Continue without feedback
                        for event in graph.stream(None, config, stream_mode="values"):
                            if "messages" in event and event["messages"]:
                                last_msg = event["messages"][-1]
                                if hasattr(last_msg, 'content') and last_msg.type == "ai":
                                    st.session_state.last_response = last_msg.content

                    # Update state based on current stage
                    current_snapshot = graph.get_state(config)

                    if st.session_state.current_stage == "planner":
                        st.session_state.outline = current_snapshot.values.get("outline")
                        st.session_state.current_stage = "writer"
                    elif st.session_state.current_stage == "writer":
                        st.session_state.blog_draft = current_snapshot.values.get("blog_draft")
                        st.session_state.current_stage = "editor"
                    elif st.session_state.current_stage == "editor":
                        st.session_state.edited_blog = current_snapshot.values.get("edited_blog")
                        st.session_state.current_stage = "social"
                    elif st.session_state.current_stage == "social":
                        st.session_state.social_posts = current_snapshot.values.get("social_posts")
                        st.session_state.current_stage = "complete"

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# -------------------------
# Stage 6: Complete
# -------------------------

elif st.session_state.current_stage == "complete":
    st.success("🎉 Blog creation complete!")

    st.markdown("### 📦 Your Deliverables")

    # Get final state
    final_snapshot = graph.get_state(config)

    # Outline
    if final_snapshot.values.get("outline"):
        with st.expander("📝 Blog Outline", expanded=False):
            st.markdown(final_snapshot.values["outline"])
            st.download_button(
                "⬇️ Download Outline",
                final_snapshot.values["outline"],
                file_name="blog_outline.txt",
                mime="text/plain"
            )

    # Draft
    if final_snapshot.values.get("blog_draft"):
        with st.expander("✍️ Blog Draft", expanded=False):
            st.markdown(final_snapshot.values["blog_draft"])
            st.download_button(
                "⬇️ Download Draft",
                final_snapshot.values["blog_draft"],
                file_name="blog_draft.md",
                mime="text/markdown"
            )

    # Edited Blog
    if final_snapshot.values.get("edited_blog"):
        with st.expander("🔍 Final Edited Blog", expanded=True):
            st.markdown(final_snapshot.values["edited_blog"])
            st.download_button(
                "⬇️ Download Final Blog",
                final_snapshot.values["edited_blog"],
                file_name="blog_final.md",
                mime="text/markdown",
                type="primary"
            )

    # Social Posts
    if final_snapshot.values.get("social_posts"):
        with st.expander("📱 Social Media Posts", expanded=False):
            social_content = final_snapshot.values["social_posts"].get("content", "")
            st.markdown(social_content)
            st.download_button(
                "⬇️ Download Social Posts",
                social_content,
                file_name="social_posts.txt",
                mime="text/plain"
            )

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Create Another Blog", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# -------------------------
# Footer
# -------------------------

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built with LangGraph, OpenAI, Tavily, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
