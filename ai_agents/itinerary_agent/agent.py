import os
import asyncio
import re
from typing import TypedDict, Annotated, List, Dict
import operator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import plugins
import requests
import json

load_dotenv()

# ============= AGENT CODE =============

# State Definition
class ItineraryState(TypedDict):
    """State for the itinerary planning agent"""
    destination: str
    duration: str
    interests: str
    budget: str
    draft_itinerary: str  # Initial day-wise breakdown for human review
    user_approved: bool  # Whether user approved the draft
    user_modifications: str  # User's changes to the draft (JSON format)
    raw_itinerary: str
    formatted_itinerary: str
    locations: List[Dict[str, any]]
    messages: Annotated[list, operator.add]


async def generate_draft_itinerary_node(state: ItineraryState) -> ItineraryState:
    """Generates initial draft itinerary for human review"""
    print("\n=== Generating Draft Itinerary ===")

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7
        )

        prompt = f"""
        You are a travel planner. Create a SIMPLE day-wise itinerary breakdown for:

        🌍 Destination: {state['destination']}
        ⏰ Duration: {state['duration']}
        🎯 Interests: {state['interests']}
        💰 Budget: {state['budget']}

        IMPORTANT: Return ONLY a JSON array with this exact structure:
        [
            {{
                "day": 1,
                "main_destination": "Area/City name for day 1",
                "places": ["Place 1", "Place 2", "Place 3"]
            }},
            {{
                "day": 2,
                "main_destination": "Area/City name for day 2",
                "places": ["Place 1", "Place 2", "Place 3"]
            }}
        ]

        Rules:
        - One object per day based on the duration
        - main_destination should be a specific area/neighborhood/city for that day
        - Include 3-5 places to visit for each day
        - Keep place names specific and searchable (e.g., "Eiffel Tower" not "famous tower")
        - Return ONLY the JSON array, no markdown, no code blocks, no extra text
        """

        response = await llm.ainvoke(prompt)
        draft_text = response.content.strip()

        # Clean up markdown if present
        if draft_text.startswith("```"):
            draft_text = draft_text.replace("```json", "").replace("```", "").strip()

        print("✅ Draft itinerary generated")

        return {**state, "draft_itinerary": draft_text, "messages": [response]}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {**state, "draft_itinerary": f"Error: {str(e)}", "messages": []}


async def generate_itinerary_node(state: ItineraryState) -> ItineraryState:
    """Generates detailed itinerary using LLM based on approved draft"""
    print("\n=== Generating Detailed Itinerary ===")

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7
        )

        # Use user modifications if available, otherwise use draft
        itinerary_plan = state.get('user_modifications') or state['draft_itinerary']

        prompt = f"""
        You are an expert travel planner. Create a detailed day-by-day itinerary based on this APPROVED plan:

        🌍 Destination: {state['destination']}
        ⏰ Duration: {state['duration']}
        🎯 Interests: {state['interests']}
        💰 Budget: {state['budget']}

        APPROVED DAY-WISE PLAN:
        {itinerary_plan}

        Create a comprehensive detailed itinerary that includes:
        1. Day-by-day breakdown with morning, afternoon, and evening activities
        2. IMPORTANT: Use the main destinations and places from the approved plan above
        3. For each place, provide detailed descriptions, timings, and tips
        4. Local food recommendations with restaurant names where possible
        5. Estimated costs for activities based on the budget level
        6. Practical tips and best times to visit each location
        7. Transportation suggestions between locations

        SPECIAL ATTENTION - Additional User Requirements:
        Check the interests section above. If it contains "IMPORTANT ADDITIONAL REQUIREMENTS (MUST INCLUDE):", these are MANDATORY user requests that MUST be incorporated into the itinerary. Find the best days and times to include these specific activities/places and add them explicitly to the detailed itinerary. Do not ignore or skip these - they are critical user requirements.

        Make it practical, engaging, and tailored to their interests and budget level.
        Format with clear sections and use emojis to make it visually appealing.

        CRITICAL CHECKLIST:
        - ✓ Follow the approved plan structure and include all the places mentioned
        - ✓ Check for "IMPORTANT ADDITIONAL REQUIREMENTS" in interests and incorporate every single one
        - ✓ Explicitly mention where and when these additional requirements are addressed in the itinerary
        """

        response = await llm.ainvoke(prompt)
        print("✅ Detailed itinerary generated")

        return {**state, "raw_itinerary": response.content, "messages": [response]}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {**state, "raw_itinerary": f"Error: {str(e)}", "messages": []}


async def extract_locations_node(state: ItineraryState) -> ItineraryState:
    """Extracts locations from itinerary and gets coordinates"""
    print("\n=== Extracting Locations ===")
    
    if "Error" in state["raw_itinerary"]:
        return {**state, "locations": []}
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )
        
        # Get the day-wise plan to understand day assignments
        day_wise_plan = state.get('user_modifications') or state['draft_itinerary']

        extraction_prompt = f"""
        From the following itinerary, extract 5-8 specific locations/attractions that can be found on a map.

        Itinerary:
        {state['raw_itinerary']}

        Day-wise Plan (for reference):
        {day_wise_plan}

        Return ONLY a JSON array with this structure:
        [
            {{"name": "Location Name, City", "day": 1}},
            {{"name": "Location Name, City", "day": 1}},
            {{"name": "Location Name, City", "day": 2}}
        ]

        Rules:
        - Each location should be specific enough to geocode
        - Assign each location to the appropriate day based on the day-wise plan
        - Return ONLY the JSON array with no markdown formatting, no code blocks, no extra text

        Example: [{{"name": "Eiffel Tower, Paris", "day": 1}}, {{"name": "Louvre Museum, Paris", "day": 2}}]
        """
        
        response = await llm.ainvoke(extraction_prompt)
        response_text = response.content.strip()
        
        # Clean up the response - remove markdown code blocks if present
        if response_text.startswith("```"):
            # Remove ```json and ``` markers
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        locations_data = json.loads(response_text)

        print(f"📍 Extracted {len(locations_data)} locations")

        # Geocode each location
        locations_with_coords = []
        for loc_data in locations_data:
            try:
                loc_name = loc_data['name']
                day = loc_data.get('day', 1)  # Default to day 1 if not specified

                # Use Nominatim for geocoding (free, no API key needed)
                url = f"https://nominatim.openstreetmap.org/search"
                params = {
                    'q': loc_name,
                    'format': 'json',
                    'limit': 1
                }
                headers = {'User-Agent': 'ItineraryPlanner/1.0'}

                response = requests.get(url, params=params, headers=headers)
                data = response.json()

                if data:
                    locations_with_coords.append({
                        'name': loc_name,
                        'lat': float(data[0]['lat']),
                        'lon': float(data[0]['lon']),
                        'display_name': data[0].get('display_name', loc_name),
                        'day': day
                    })
                    print(f"✅ Geocoded: {loc_name} (Day {day})")
                else:
                    print(f"⚠️ Could not geocode: {loc_name}")
                    
            except Exception as e:
                print(f"❌ Error geocoding {loc_name}: {e}")
                continue
        
        print(f"✅ Successfully geocoded {len(locations_with_coords)} locations")
        
        return {**state, "locations": locations_with_coords}
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Response was: {response_text[:200]}...")
        # Fallback: return empty locations
        return {**state, "locations": []}
        
    except Exception as e:
        print(f"❌ Error extracting locations: {e}")
        return {**state, "locations": []}


async def format_output_node(state: ItineraryState) -> ItineraryState:
    """Formats and structures the final itinerary output"""
    print("\n=== Formatting Output ===")
    
    if "Error" in state["raw_itinerary"]:
        return {**state, "formatted_itinerary": state["raw_itinerary"]}
    
    try:
        formatted = f"""
# 🗺️ Your Personalized Travel Itinerary

**Destination:** {state['destination']}
**Duration:** {state['duration']}
**Budget Level:** {state['budget']}
**Interests:** {state['interests']}

---

{state['raw_itinerary']}

---

## 📍 Places to Visit ({len(state['locations'])} locations plotted on map)

{chr(10).join([f"• {loc['name']}" for loc in state['locations']])}

---

## 💡 Quick Tips
- Book accommodations and major attractions in advance
- Keep digital and physical copies of important documents
- Download offline maps for your destination
- Check visa requirements and travel advisories
- Consider travel insurance for peace of mind

**Have an amazing trip! 🌟**
"""
        
        print("✅ Output formatted")
        return {**state, "formatted_itinerary": formatted}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {**state, "formatted_itinerary": f"Error: {str(e)}"}


def create_draft_graph():
    """Creates workflow for generating draft itinerary only"""
    workflow = StateGraph(ItineraryState)

    workflow.add_node("generate_draft", generate_draft_itinerary_node)

    workflow.set_entry_point("generate_draft")
    workflow.add_edge("generate_draft", END)

    return workflow.compile()


def create_final_graph():
    """Creates workflow for generating final itinerary after approval"""
    workflow = StateGraph(ItineraryState)

    workflow.add_node("generate_itinerary", generate_itinerary_node)
    workflow.add_node("extract_locations", extract_locations_node)
    workflow.add_node("format_output", format_output_node)

    workflow.set_entry_point("generate_itinerary")
    workflow.add_edge("generate_itinerary", "extract_locations")
    workflow.add_edge("extract_locations", "format_output")
    workflow.add_edge("format_output", END)

    return workflow.compile()


async def run_draft_agent(destination: str, duration: str, interests: str, budget: str):
    """Generates draft itinerary for human review"""
    graph = create_draft_graph()

    initial_state = {
        "destination": destination,
        "duration": duration,
        "interests": interests,
        "budget": budget,
        "draft_itinerary": "",
        "user_approved": False,
        "user_modifications": "",
        "raw_itinerary": "",
        "formatted_itinerary": "",
        "locations": [],
        "messages": [],
    }

    result = await graph.ainvoke(initial_state)
    return result["draft_itinerary"]


async def run_final_agent(destination: str, duration: str, interests: str, budget: str, user_modifications: str, anything_else: str = ""):
    """Generates final detailed itinerary based on approved draft"""
    graph = create_final_graph()

    # Append anything_else to interests if provided with more emphasis
    enhanced_interests = interests
    if anything_else.strip():
        enhanced_interests = f"{interests}.\n\nIMPORTANT ADDITIONAL REQUIREMENTS (MUST INCLUDE): {anything_else}"

    initial_state = {
        "destination": destination,
        "duration": duration,
        "interests": enhanced_interests,
        "budget": budget,
        "draft_itinerary": "",
        "user_approved": True,
        "user_modifications": user_modifications,
        "raw_itinerary": "",
        "formatted_itinerary": "",
        "locations": [],
        "messages": [],
    }

    result = await graph.ainvoke(initial_state)
    return result["formatted_itinerary"], result["locations"]


# ============= MAP FUNCTIONS =============

def create_selection_map():
    """Creates an interactive map for country selection"""
    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles="OpenStreetMap"
    )
    
    # Add click functionality info
    folium.Marker(
        [0, 0],
        popup="Click anywhere on the map to select a location",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    
    m.add_child(folium.LatLngPopup())
    
    return m


def create_itinerary_map(locations: List[Dict], center_location: str):
    """Creates a map with itinerary locations and routes with day-wise colors and filtering"""

    if not locations:
        # Default map
        m = folium.Map(location=[20, 0], zoom_start=2)
        return m

    # Calculate center
    avg_lat = sum(loc['lat'] for loc in locations) / len(locations)
    avg_lon = sum(loc['lon'] for loc in locations) / len(locations)

    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    # Define colors for different days (supports up to 10 days)
    day_colors = {
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'purple',
        5: 'orange',
        6: 'darkred',
        7: 'lightred',
        8: 'beige',
        9: 'darkblue',
        10: 'darkgreen'
    }

    # Group locations by day
    locations_by_day = {}
    for loc in locations:
        day = loc.get('day', 1)
        if day not in locations_by_day:
            locations_by_day[day] = []
        locations_by_day[day].append(loc)

    # Create a feature group for each day
    for day in sorted(locations_by_day.keys()):
        day_locs = locations_by_day[day]
        color = day_colors.get(day, 'gray')

        # Create feature group for this day (for layer control)
        feature_group = folium.FeatureGroup(name=f'Day {day}', show=True)

        # Add markers for this day
        for idx, loc in enumerate(day_locs, 1):
            # Main marker
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=f"<b>Day {day} - Stop {idx}</b><br>{loc['name']}<br>{loc['display_name']}",
                tooltip=f"Day {day}: {loc['name']}",
                icon=folium.Icon(color=color, icon='info-sign', prefix='fa')
            ).add_to(feature_group)

            # Numbered circle marker
            folium.CircleMarker(
                location=[loc['lat'], loc['lon']],
                radius=15,
                popup=f"Day {day} - {loc['name']}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(feature_group)

        # Draw route lines between locations of this day
        if len(day_locs) > 1:
            coordinates = [[loc['lat'], loc['lon']] for loc in day_locs]

            folium.PolyLine(
                coordinates,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f'Day {day} Route'
            ).add_to(feature_group)

            # Add arrow decorators
            plugins.AntPath(
                coordinates,
                color=color,
                weight=3,
                opacity=0.6,
                delay=1000
            ).add_to(feature_group)

        # Add the feature group to the map
        feature_group.add_to(m)

    # Add layer control for day filtering
    folium.LayerControl(collapsed=False).add_to(m)

    # Fit bounds to show all markers
    if locations:
        bounds = [[loc['lat'], loc['lon']] for loc in locations]
        m.fit_bounds(bounds, padding=(30, 30))

    return m


def reverse_geocode(lat: float, lon: float) -> str:
    """Get location name from coordinates"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json'
        }
        headers = {'User-Agent': 'ItineraryPlanner/1.0'}
        
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        # Try to get country or city
        address = data.get('address', {})
        location = address.get('country', '') or address.get('city', '') or data.get('display_name', '')
        
        return location
        
    except Exception as e:
        print(f"Geocoding error: {e}")
        return f"{lat:.2f}, {lon:.2f}"


# ============= STREAMLIT UI =============

st.set_page_config(
    page_title="AI Itinerary Planner with Map",
    page_icon="🗺️",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🗺️ AI Itinerary Planner</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter details → Review & edit draft → Generate personalized itinerary → See routes on map</p>", unsafe_allow_html=True)

# Sidebar
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
    🛣️ Route visualization
    💰 Budget-conscious
    📥 Download itinerary
    """)

# Initialize session state
# Initialize session state - ONLY ONCE
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

if 'last_result' not in st.session_state:
    st.session_state.last_result = None

if 'locations' not in st.session_state:
    st.session_state.locations = []

if 'selected_location' not in st.session_state:
    st.session_state.selected_location = ""

# New session state for human-in-the-loop workflow
if 'workflow_stage' not in st.session_state:
    st.session_state.workflow_stage = "input"  # Stages: input, draft_review, final

if 'draft_data' not in st.session_state:
    st.session_state.draft_data = None

if 'trip_params' not in st.session_state:
    st.session_state.trip_params = {}

if 'anything_else' not in st.session_state:
    st.session_state.anything_else = ""


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

    # Button callback
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

        # Buttons
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

        dest_name = st.session_state.trip_params.get("destination", "trip").replace(' ', '_').replace(',', '')
        st.download_button(
            label="📥 Download Itinerary",
            data=st.session_state.last_result,
            file_name=f"itinerary_{dest_name}.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Add a reset button
        def reset_workflow():
            st.session_state.workflow_stage = "input"
            st.session_state.last_result = None
            st.session_state.locations = []
            st.session_state.draft_data = None
            st.session_state.trip_params = {}
            st.session_state.anything_else = ""

        if st.button("🔄 Plan Another Trip", use_container_width=True, on_click=reset_workflow):
            pass

    with col2:
        st.markdown("### 🗺️ Your Route Map")

        if st.session_state.locations:
            st.info(f"📍 {len(st.session_state.locations)} locations plotted with route")
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

# Help section - show only in input stage
if st.session_state.workflow_stage == "input":
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

st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using LangGraph, GPT-4o, Streamlit & Folium</p>", unsafe_allow_html=True)