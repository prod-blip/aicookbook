"""Draft itinerary agent for human-in-the-loop workflow"""
import json
from typing import TypedDict, Annotated, List, Dict
import operator
from langgraph.graph import StateGraph, END
from utils.cleanup import strip_json
from utils.llm import get_llm


class DraftState(TypedDict):
    """State for draft itinerary generation"""
    destination: str
    duration: str
    interests: str
    budget: str
    draft_itinerary: str
    messages: Annotated[list, operator.add]


async def generate_draft_itinerary_node(state: DraftState) -> DraftState:
    """
    Generates initial draft itinerary for human review.
    
    Returns a JSON array with day-wise breakdown that users can edit.
    """
    print("\n=== Generating Draft Itinerary ===")

    try:
        llm = get_llm(temperature=0.7)

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
        cleaned_text = strip_json(response.content)

        print("✅ Draft itinerary generated")

        return {
            **state,
            "draft_itinerary": cleaned_text,
            "messages": [response]
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            **state,
            "draft_itinerary": f"Error: {str(e)}",
            "messages": []
        }


def create_draft_graph():
    """Creates and compiles the draft generation workflow"""
    workflow = StateGraph(DraftState)

    workflow.add_node("generate_draft", generate_draft_itinerary_node)

    workflow.set_entry_point("generate_draft")
    workflow.add_edge("generate_draft", END)

    return workflow.compile()


async def run_draft_agent(destination: str, duration: str, interests: str, budget: str) -> str:
    """
    Main function to run the draft generation agent.
    
    Args:
        destination: Travel destination
        duration: Trip duration (e.g., "5 days")
        interests: User interests
        budget: Budget level (Low/Medium/High)
        
    Returns:
        Draft itinerary as JSON string
    """
    graph = create_draft_graph()

    initial_state = {
        "destination": destination,
        "duration": duration,
        "interests": interests,
        "budget": budget,
        "draft_itinerary": "",
        "messages": [],
    }

    result = await graph.ainvoke(initial_state)
    return result["draft_itinerary"]