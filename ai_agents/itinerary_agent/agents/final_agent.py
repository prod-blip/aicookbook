"""Final itinerary agent with location extraction"""
import json
from typing import TypedDict, Annotated, List, Dict
import operator
from langgraph.graph import StateGraph, END
from utils.cleanup import strip_json
from utils.geo import geocode
from utils.llm import get_llm


class FinalState(TypedDict):
    """State for final itinerary generation"""
    destination: str
    duration: str
    interests: str
    budget: str
    user_modifications: str
    raw_itinerary: str
    formatted_itinerary: str
    locations: List[Dict[str, any]]
    messages: Annotated[list, operator.add]


async def generate_itinerary_node(state: FinalState) -> FinalState:
    """
    Generates detailed itinerary using LLM based on approved draft.
    
    Incorporates user modifications and additional requirements.
    """
    print("\n=== Generating Detailed Itinerary ===")

    try:
        llm = get_llm(temperature=0.7)

        # Use user modifications
        itinerary_plan = state['user_modifications']

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

        return {
            **state,
            "raw_itinerary": response.content,
            "messages": [response]
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            **state,
            "raw_itinerary": f"Error: {str(e)}",
            "messages": []
        }


async def extract_locations_node(state: FinalState) -> FinalState:
    """
    Extracts locations from itinerary and gets coordinates.
    
    Uses LLM to extract specific locations and assigns them to days,
    then geocodes each location using Nominatim.
    """
    print("\n=== Extracting Locations ===")
    
    if "Error" in state["raw_itinerary"]:
        return {**state, "locations": []}
    
    try:
        llm = get_llm(temperature=0)
        
        # Get the day-wise plan to understand day assignments
        day_wise_plan = state['user_modifications']

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
        cleaned_text = strip_json(response.content)
        
        # Parse JSON
        locations_data = json.loads(cleaned_text)

        print(f"📍 Extracted {len(locations_data)} locations")

        # Geocode each location
        locations_with_coords = []
        for loc_data in locations_data:
            try:
                loc_name = loc_data['name']
                day = loc_data.get('day', 1)

                # Geocode using helper function
                coords = geocode(loc_name)

                if coords:
                    locations_with_coords.append({
                        'name': loc_name,
                        'day': day,
                        **coords
                    })
                    print(f"✅ Geocoded: {loc_name} (Day {day})")
                else:
                    print(f"⚠️ Could not geocode: {loc_name}")
                    
            except Exception as e:
                print(f"❌ Error processing {loc_name}: {e}")
                continue
        
        print(f"✅ Successfully geocoded {len(locations_with_coords)} locations")
        
        return {**state, "locations": locations_with_coords}
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Response was: {cleaned_text[:200]}...")
        return {**state, "locations": []}
        
    except Exception as e:
        print(f"❌ Error extracting locations: {e}")
        return {**state, "locations": []}


async def format_output_node(state: FinalState) -> FinalState:
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


def create_final_graph():
    """Creates and compiles the final itinerary workflow"""
    workflow = StateGraph(FinalState)

    workflow.add_node("generate_itinerary", generate_itinerary_node)
    workflow.add_node("extract_locations", extract_locations_node)
    workflow.add_node("format_output", format_output_node)

    workflow.set_entry_point("generate_itinerary")
    workflow.add_edge("generate_itinerary", "extract_locations")
    workflow.add_edge("extract_locations", "format_output")
    workflow.add_edge("format_output", END)

    return workflow.compile()


async def run_final_agent(
    destination: str,
    duration: str,
    interests: str,
    budget: str,
    user_modifications: str,
    anything_else: str = ""
) -> tuple:
    """
    Main function to run the final itinerary agent.
    
    Args:
        destination: Travel destination
        duration: Trip duration
        interests: User interests
        budget: Budget level
        user_modifications: Approved draft with user edits (JSON string)
        anything_else: Additional requirements from user
        
    Returns:
        Tuple of (formatted_itinerary, locations)
    """
    graph = create_final_graph()

    # Append anything_else to interests if provided with emphasis
    enhanced_interests = interests
    if anything_else.strip():
        enhanced_interests = f"{interests}.\n\nIMPORTANT ADDITIONAL REQUIREMENTS (MUST INCLUDE): {anything_else}"

    initial_state = {
        "destination": destination,
        "duration": duration,
        "interests": enhanced_interests,
        "budget": budget,
        "user_modifications": user_modifications,
        "raw_itinerary": "",
        "formatted_itinerary": "",
        "locations": [],
        "messages": [],
    }

    result = await graph.ainvoke(initial_state)
    return result["formatted_itinerary"], result["locations"]