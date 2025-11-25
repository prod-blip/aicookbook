"""
AI Itinerary Planner - Streamlit Application

Human-in-the-loop travel itinerary planning with interactive maps.
"""
import os
import asyncio
import json
import streamlit as st
from streamlit_folium import st_folium
from dotenv import load_dotenv

from agents.draft_agent import run_draft_agent
from agents.final_agent import run_final_agent
from maps.map_utils import create_itinerary_map

load_dotenv()


# ============= SESSION STATE INITIALIZATION =============

def init_session_state():
    """Initialize all session state variables"""
    
    # Event loop - ONLY ONCE
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)

    # Processing flags
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    # Results
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    if 'locations' not in st.session_state:
        st.session_state.locations = []

    # Workflow state
    if 'workflow_stage' not in st.session_state:
        st.session_state.workflow_stage = "input"

    if 'draft_data' not in st.session_state:
        st.session_state.draft_data = None

    if 'trip_params' not in st.session_state:
        st.session_state.trip_params = {}

    if 'anything_else' not in st.session_state:
        st.session_state.anything_else = ""


# ============= PAGE CONFIGURATION =============

st.set_page_config(
    page_title="Itinerary Planner Agent",
    page_icon="🗺️",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>🗺️ Itinerary Planner Agent</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Enter details → Review & edit draft → Generate personalized itinerary → See routes on map</p>",
    unsafe_allow_html=True
)


# ============= SIDEBAR =============

with st.sidebar:
    st.markdown("### 🔑 Configuration Status")
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    
    st.markdown(f"{'✅' if has_api_key else '❌'} OpenAI API Key")
    
    if not has_api_key:
        st.warning("⚠️ Please set OPENAI_API_KEY in .env file")
    
    st.markdown("---")
    st.markdown("### 📋 How It Works")
    st.markdown("""
    1. **Enter trip details** (destination, duration, interests, budget)
    2. **Review draft** - AI generates day-wise plan
    3. **Edit & approve** - Modify destinations and places
    4. **Get full itinerary** with map and routes
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown("""
    👤 Human-in-the-loop editing
    ✈️ Day-by-day itinerary
    📍 Auto-plot destinations
    🗺️ Interactive map with routes
    🎨 Day-wise color coding
    💰 Budget-conscious
    📥 Download itinerary
    """)


# ============= INITIALIZE SESSION STATE =============

init_session_state()


# ============= STAGE 1: INPUT FORM =============

if st.session_state.workflow_stage == "input":
    st.markdown("### 📝 Step 1: Enter Trip Details")

    col1, col2 = st.columns(2)

    with col1:
        destination = st.text_input(
            "🌍 Destination",
            placeholder="e.g., Paris, France",
            help="Enter your travel destination",
            key="destination_input"
        )

        duration = st.text_input(
            "⏰ Duration",
            placeholder="e.g., 5 days",
            help="How long is your trip?",
            key="duration_input"
        )

    with col2:
        interests = st.text_input(
            "🎯 Interests",
            placeholder="e.g., art, food, history",
            help="What are you interested in?",
            key="interests_input"
        )

        budget = st.selectbox(
            "💰 Budget Level",
            options=["Low", "Medium", "High"],
            index=1,
            key="budget_input"
        )

    # Validation and draft generation
    def start_draft_generation():
        if not destination or not duration or not interests:
            st.error("❌ Please fill in all fields")
            return
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OpenAI API key not found in .env file")
            return
        
        st.session_state.is_processing = True
        st.session_state.trip_params = {
            "destination": destination,
            "duration": duration,
            "interests": interests,
            "budget": budget
        }

    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button(
            "📝 Generate Draft Itinerary",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_processing,
            on_click=start_draft_generation,
            key="generate_draft_button"
        )

    # Process draft generation
    if st.session_state.is_processing:
        with st.spinner("🔄 Creating draft itinerary..."):
            st.write("⏳ Analyzing your requirements...")
            st.write("🤖 Generating day-wise breakdown...")

            draft = st.session_state.loop.run_until_complete(
                run_draft_agent(
                    st.session_state.trip_params["destination"],
                    st.session_state.trip_params["duration"],
                    st.session_state.trip_params["interests"],
                    st.session_state.trip_params["budget"]
                )
            )

        st.session_state.draft_data = draft
        st.session_state.workflow_stage = "draft_review"
        st.session_state.is_processing = False
        st.rerun()

    # Help section for input stage
    st.markdown("---")
    st.markdown("""
        <div style='padding: 25px; background-color: #f0f2f6; border-radius: 15px;'>
        <h3>🚀 Getting Started</h3>
        <ol>
            <li><strong>Enter destination:</strong> Type your travel destination</li>
            <li><strong>Enter details:</strong> Duration, interests, and budget</li>
            <li><strong>Review draft:</strong> AI creates a day-wise plan for you to review and edit</li>
            <li><strong>Approve & generate:</strong> Get detailed itinerary with map routes!</li>
        </ol>
        <p><strong>Example:</strong> Bali, Indonesia → 5 days → beaches, temples, nature → Medium budget</p>
        <p><em>You'll have full control to modify destinations and places before generating the final itinerary!</em></p>
        </div>
    """, unsafe_allow_html=True)


# ============= STAGE 2: DRAFT REVIEW =============

elif st.session_state.workflow_stage == "draft_review":
    st.markdown("### ✏️ Step 2: Review and Edit Draft Itinerary")
    st.info("👇 Review the day-wise plan below. You can edit the main destination and places for each day.")

    try:
        draft_json = json.loads(st.session_state.draft_data)

        # Store edited data
        edited_draft = []

        for day_data in draft_json:
            st.markdown(f"#### 📅 Day {day_data['day']}")

            col1, col2 = st.columns([1, 2])

            with col1:
                main_dest = st.text_input(
                    "Main Destination",
                    value=day_data['main_destination'],
                    key=f"main_dest_day_{day_data['day']}"
                )

            with col2:
                places_str = ", ".join(day_data['places'])
                places = st.text_area(
                    "Places to Visit (comma-separated)",
                    value=places_str,
                    key=f"places_day_{day_data['day']}",
                    height=100
                )

            # Store edited data
            edited_draft.append({
                "day": day_data['day'],
                "main_destination": main_dest,
                "places": [p.strip() for p in places.split(",") if p.strip()]
            })

            st.markdown("---")

        # Anything Else section
        st.markdown("### 📝 Anything Else?")
        anything_else = st.text_area(
            "Add any specific places, requirements, or preferences you'd like to include",
            placeholder="e.g., I want to visit a local night market, try authentic street food, visit a specific temple...",
            help="These will be incorporated into your final itinerary in the best possible way",
            key="anything_else_input",
            height=100
        )

        st.markdown("---")

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        def approve_and_generate():
            st.session_state.is_processing = True
            st.session_state.edited_draft = json.dumps(edited_draft)
            st.session_state.anything_else = anything_else

        def go_back():
            st.session_state.workflow_stage = "input"
            st.session_state.draft_data = None

        with col1:
            st.button(
                "⬅️ Back to Edit Details",
                use_container_width=True,
                on_click=go_back
            )

        with col3:
            st.button(
                "✅ Approve & Generate Full Itinerary",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_processing,
                on_click=approve_and_generate,
                key="approve_button"
            )

        # Process final generation
        if st.session_state.is_processing:
            with st.spinner("🔄 Generating detailed itinerary..."):
                st.write("⏳ Processing approved plan...")
                st.write("🤖 Creating detailed day-by-day itinerary...")
                st.write("📍 Extracting and geocoding locations...")
                st.write("🗺️ Creating interactive map with routes...")

                itinerary, locations = st.session_state.loop.run_until_complete(
                    run_final_agent(
                        st.session_state.trip_params["destination"],
                        st.session_state.trip_params["duration"],
                        st.session_state.trip_params["interests"],
                        st.session_state.trip_params["budget"],
                        st.session_state.edited_draft,
                        st.session_state.get("anything_else", "")
                    )
                )

            st.session_state.last_result = itinerary
            st.session_state.locations = locations
            st.session_state.workflow_stage = "final"
            st.session_state.is_processing = False
            st.rerun()

    except json.JSONDecodeError:
        st.error("❌ Error parsing draft itinerary. Please try generating again.")
        if st.button("🔄 Start Over"):
            st.session_state.workflow_stage = "input"
            st.session_state.draft_data = None
            st.rerun()


# ============= STAGE 3: FINAL RESULTS =============

elif st.session_state.workflow_stage == "final" and st.session_state.last_result:
    st.markdown("---")
    st.markdown("## 🎉 Your Personalized Itinerary")

    # Two columns - itinerary and map
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Itinerary Details")
        with st.container():
            st.markdown(st.session_state.last_result)

        # Download button
        dest_name = st.session_state.trip_params.get("destination", "trip").replace(' ', '_').replace(',', '')
        st.download_button(
            label="📥 Download Itinerary",
            data=st.session_state.last_result,
            file_name=f"itinerary_{dest_name}.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Reset button
        def reset_workflow():
            st.session_state.workflow_stage = "input"
            st.session_state.last_result = None
            st.session_state.locations = []
            st.session_state.draft_data = None
            st.session_state.trip_params = {}
            st.session_state.anything_else = ""

        st.button(
            "🔄 Plan Another Trip",
            use_container_width=True,
            on_click=reset_workflow
        )

    with col2:
        st.markdown("### 🗺️ Your Route Map")

        if st.session_state.locations:
            st.info(f"📍 {len(st.session_state.locations)} locations plotted with routes")
            
            dest_for_map = st.session_state.trip_params.get("destination", "")
            itinerary_map = create_itinerary_map(st.session_state.locations, dest_for_map)
            
            st_folium(
                itinerary_map,
                width=700,
                height=600,
                key=f"itinerary_map_{len(st.session_state.locations)}"
            )
        else:
            st.warning("⚠️ No locations were extracted. Try generating again.")


# ============= FOOTER =============

st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ using LangGraph, GPT-4o, Streamlit & Folium</p>",
    unsafe_allow_html=True
)