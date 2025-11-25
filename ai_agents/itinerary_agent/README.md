# AI Itinerary Planner

An intelligent travel planning assistant that generates personalized day-by-day itineraries with interactive route maps, featuring a human-in-the-loop workflow for complete user control over destinations and activities.

✨ **Powered by LangGraph multi-agent architecture with GPT-4o and interactive Folium maps**

## Features

* **Human-in-the-Loop Workflow** - Review and edit AI-generated plans before final generation, ensuring complete control over your itinerary
* **Two-Stage Agent System** - Draft agent creates editable day-wise breakdown, Final agent generates comprehensive detailed itinerary
* **Intelligent Location Extraction** - AI automatically identifies and geocodes 5-8 key destinations from your itinerary text
* **Interactive Day-Wise Maps** - Color-coded routes for each day with layer controls to filter and focus on specific days
* **Flexible Customization** - "Anything Else" field lets you add must-visit places or special requirements that get incorporated automatically
* **Budget-Aware Recommendations** - Get suggestions tailored to Low, Medium, or High budget levels
* **Downloadable Itineraries** - Export your complete travel plan as Markdown for offline access
* **Free Geocoding & Maps** - Uses OpenStreetMap and Nominatim, no Mapbox API required

## Setup

### Requirements

* Python 3.8+
* OpenAI API Key (for GPT-4o)
* Internet connection (for geocoding and map tiles)

### Installation

1. Clone this repository or download the `optimized/` folder:

```bash
cd optimized/
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

Packages installed:
- `langgraph` - Agent orchestration framework
- `langchain-openai` - OpenAI GPT-4o integration
- `streamlit` - Web interface
- `streamlit-folium` - Interactive maps in Streamlit
- `folium` - Map visualization library
- `requests` - API calls for geocoding
- `python-dotenv` - Environment variable management

3. Get your API credentials:
   * **OpenAI API Key**: https://platform.openai.com/api-keys
     - Sign up or log in to OpenAI
     - Navigate to API Keys section
     - Create a new secret key
     - Copy and save securely

4. Setup your `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-proj-...your-actual-key-here
```

**Important:** Never commit the `.env` file to version control. It's included in `.gitignore`.

## Running the App

1. Navigate to the optimized directory:

```bash
cd optimized/
```

2. Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

3. Your browser will automatically open to `http://localhost:8501`

4. If the browser doesn't open automatically, manually navigate to the URL shown in the terminal

5. You'll see the three-stage workflow interface ready to use

## How It Works - Complete Workflow

### Stage 1: Input Trip Details

**What you do:**
- Enter your destination (e.g., "Tokyo, Japan")
- Specify duration (e.g., "7 days", "1 week")
- Add interests (e.g., "anime, food, temples, technology")
- Select budget level (Low, Medium, High)
- Click "Generate Draft Itinerary"

**What happens:**
- Draft Agent activates
- GPT-4o analyzes your requirements
- Creates a day-by-day breakdown in JSON format
- Each day gets a main destination and 3-5 places to visit
- Takes 5-10 seconds

### Stage 2: Review & Edit Draft

**What you see:**
- Day-by-day breakdown with editable fields
- Main destination for each day (text input)
- Places to visit for each day (comma-separated text area)
- "Anything Else" field for additional requirements

**What you do:**
- Review the AI-generated plan
- Edit main destinations if needed (e.g., change "Shibuya" to "Akihabara")
- Modify places to visit (add, remove, or reorder)
- Add any missed requirements in "Anything Else" (e.g., "I want to visit teamLab Borderless, try authentic ramen at Ichiran")
- Click "Approve & Generate Full Itinerary"

**Why this matters:**
- You have complete control before generating the detailed plan
- Ensures AI includes your must-visit places
- Prevents wasted API calls on unwanted suggestions

### Stage 3: View Final Results

**What you get:**

**Left Side - Detailed Itinerary:**
- Complete day-by-day breakdown with timings
- Morning, afternoon, and evening activities
- Specific place descriptions and tips
- Local food recommendations with restaurant names
- Estimated costs based on your budget
- Transportation suggestions between locations
- Practical tips for each destination

**Right Side - Interactive Map:**
- All locations plotted with numbered markers
- Color-coded by day:
  - Day 1: Red
  - Day 2: Blue  
  - Day 3: Green
  - Day 4: Purple
  - Day 5: Orange
  - (Up to 10 days with unique colors)
- Routes connecting locations within each day
- Animated ant paths showing direction
- Layer control checkboxes to show/hide specific days
- Click markers for location details
- Auto-zoomed to show all destinations

**Actions available:**
- Download itinerary as Markdown file
- Plan another trip (resets workflow)

## Agent Architecture

The application uses **LangGraph** with a **two-stage agentic workflow**:

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 1: DRAFT GENERATION                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  User Inputs:    │
                  │  • Destination   │
                  │  • Duration      │
                  │  • Interests     │
                  │  • Budget        │
                  └────────┬─────────┘
                           │
                           ▼
        ╔══════════════════════════════════════╗
        ║     DRAFT AGENT (LangGraph)          ║
        ╠══════════════════════════════════════╣
        ║  Node: generate_draft_itinerary      ║
        ║  • Analyzes requirements             ║
        ║  • Creates day-wise JSON structure   ║
        ║  • Assigns main destination per day  ║
        ║  • Suggests 3-5 places per day       ║
        ╚══════════════════════════════════════╝
                           │
                           ▼
                  ┌──────────────────┐
                  │  Draft Output:   │
                  │  JSON array with │
                  │  day-wise plan   │
                  └────────┬─────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2: HUMAN REVIEW & EDITING                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  User Reviews:   │
                  │  • Edits days    │
                  │  • Changes places│
                  │  • Adds extras   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Approved Draft  │
                  │  + "Anything     │
                  │     Else" reqs   │
                  └────────┬─────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 3: FINAL GENERATION                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
        ╔══════════════════════════════════════╗
        ║      FINAL AGENT (LangGraph)         ║
        ╠══════════════════════════════════════╣
        ║  Node 1: generate_itinerary          ║
        ║  • Takes approved draft as input     ║
        ║  • Incorporates "Anything Else"      ║
        ║  • Generates detailed descriptions   ║
        ║  • Adds timings and costs            ║
        ║  • Includes transport suggestions    ║
        ╠══════════════════════════════════════╣
        ║  Node 2: extract_locations           ║
        ║  • AI reads generated itinerary      ║
        ║  • Identifies 5-8 key locations      ║
        ║  • Assigns each to correct day       ║
        ║  • Geocodes using Nominatim API      ║
        ║  • Returns lat/lon coordinates       ║
        ╠══════════════════════════════════════╣
        ║  Node 3: format_output               ║
        ║  • Structures final markdown         ║
        ║  • Adds headers and sections         ║
        ║  • Lists all plotted locations       ║
        ║  • Includes travel tips              ║
        ╚══════════════════════════════════════╝
                           │
                           ├──────────────┬─────────────┐
                           ▼              ▼             ▼
                  ┌──────────────┐  ┌──────────┐  ┌────────┐
                  │   Formatted  │  │Locations │  │  Map   │
                  │   Itinerary  │  │with coords│  │ Data   │
                  └──────────────┘  └──────────┘  └────────┘
                                          │
                                          ▼
                              ┌────────────────────┐
                              │  create_itinerary_ │
                              │  map() Function    │
                              │  • Groups by day   │
                              │  • Applies colors  │
                              │  • Adds markers    │
                              │  • Draws routes    │
                              │  • Layer controls  │
                              └────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────┐
                              │  Interactive Map   │
                              │  Displayed in UI   │
                              └────────────────────┘
```

### Node Descriptions

**Draft Agent (Single Node):**

1. **generate_draft_itinerary_node**
   - **Purpose**: Create a simple, editable day-wise plan
   - **Input**: Destination, duration, interests, budget
   - **LLM Call**: GPT-4o with structured JSON prompt
   - **Output**: JSON array with day, main_destination, and places for each day
   - **Processing Time**: 5-10 seconds

**Final Agent (Three Sequential Nodes):**

1. **generate_itinerary_node**
   - **Purpose**: Create comprehensive detailed itinerary
   - **Input**: Approved draft + original parameters + "Anything Else" requirements
   - **Special Processing**: Looks for "IMPORTANT ADDITIONAL REQUIREMENTS" flag in interests
   - **LLM Call**: GPT-4o with detailed prompt including approved plan
   - **Output**: Rich markdown text with full day-by-day breakdown
   - **Processing Time**: 15-30 seconds

2. **extract_locations_node**
   - **Purpose**: Identify and geocode key destinations
   - **Input**: Generated itinerary text + day-wise plan for reference
   - **LLM Call**: GPT-4o with extraction prompt (temperature=0 for consistency)
   - **Geocoding**: Nominatim API (OpenStreetMap) for each location
   - **Output**: List of locations with name, lat, lon, display_name, and day assignment
   - **Error Handling**: Skips locations that can't be geocoded
   - **Processing Time**: 10-20 seconds (includes geocoding API calls)

3. **format_output_node**
   - **Purpose**: Structure the final markdown output
   - **Input**: Raw itinerary + locations list
   - **Processing**: Adds headers, metadata, location list, travel tips
   - **Output**: Complete formatted markdown ready for display/download
   - **Processing Time**: <1 second

### Data Flow

```
User Input → Draft JSON → User Edits → Approved JSON → Detailed Text → Locations Array → Formatted Output + Map
```

### State Management

Both agents use **TypedDict** for type-safe state management:

**DraftState:**
```python
{
    "destination": str,
    "duration": str,
    "interests": str,
    "budget": str,
    "draft_itinerary": str,  # JSON string
    "messages": list
}
```

**FinalState:**
```python
{
    "destination": str,
    "duration": str,
    "interests": str,
    "budget": str,
    "user_modifications": str,  # Approved JSON
    "raw_itinerary": str,
    "formatted_itinerary": str,
    "locations": List[Dict],  # Geocoded locations
    "messages": list
}
```

## Key Features Explained

### Human-in-the-Loop Design

**Problem Solved:** AI can sometimes suggest places you don't want to visit or miss your must-see destinations.

**Solution:** Two-stage generation with human approval in between ensures the final detailed itinerary is based on YOUR edited plan, not just the AI's initial suggestion.

### Intelligent Location Extraction

**How it works:**
1. Final Agent generates detailed text itinerary
2. A separate LLM call extracts 5-8 specific location names
3. Each location is assigned to the correct day based on the approved plan
4. Nominatim API geocodes each location (free, no API key needed)
5. Failed geocoding requests are skipped (logged but don't break the flow)

**Why this approach:**
- More reliable than regex parsing
- Handles various location name formats
- Day assignment ensures correct color coding on map

### Day-Wise Color-Coded Maps

**Color Palette:**
- Day 1: Red
- Day 2: Blue
- Day 3: Green
- Day 4: Purple
- Day 5: Orange
- Day 6: Dark Red
- Day 7: Light Red
- Day 8: Beige
- Day 9: Dark Blue
- Day 10: Dark Green

**Map Features:**
- **Markers**: Numbered icons showing stop sequence
- **Routes**: Polylines connecting locations within each day
- **Animated Paths**: Ant path animation showing route direction
- **Layer Control**: Checkboxes to toggle day visibility
- **Popups**: Click markers for location details
- **Auto-fit**: Map automatically zooms to show all markers

### "Anything Else" Feature

**Purpose:** Capture requirements that might have been missed in initial planning.

**How it works:**
- User adds text like: "I want to visit the Ghibli Museum and try authentic okonomiyaki"
- System prepends "IMPORTANT ADDITIONAL REQUIREMENTS (MUST INCLUDE):" to this text
- Final Agent's prompt explicitly instructs it to incorporate ALL these requirements
- LLM is told these are MANDATORY and must be explicitly mentioned in the itinerary

**Why the emphasis:** Without strong prompting, LLMs sometimes skip user-added requirements. The MUST INCLUDE flag ensures they're prioritized.

## Important Notes

⚠️ **Geocoding Rate Limits**: Nominatim has a rate limit of 1 request per second. The app handles this automatically, but extracting many locations may take 10-20 seconds.

🔐 **API Key Security**: Never share your OpenAI API key. Keep it in `.env` and ensure `.env` is in `.gitignore`. The key is only used server-side (not sent to browser).

💰 **Cost Considerations**: 
- Each itinerary uses 2-3 GPT-4o API calls
- Estimated cost: $0.02-$0.05 per itinerary
- Draft generation: ~$0.01
- Final generation: ~$0.02-$0.03
- Location extraction: ~$0.01
- Monitor your usage at https://platform.openai.com/usage

📊 **Data Handling**: 
- No data is stored permanently
- All data lives in Streamlit session state
- Closing the browser tab clears all data
- Downloaded files are only stored locally on your machine

🗺️ **Map Limitations**:
- Routes shown are straight lines, not actual roads
- For real road routing, consider adding OpenRouteService API (free tier available)
- Some obscure locations may not geocode successfully
- Internet connection required for map tiles

⏱️ **Processing Time**:
- Draft generation: 5-10 seconds
- Final generation: 15-30 seconds  
- Location extraction: 10-20 seconds
- Total: 30-60 seconds for complete itinerary

## Troubleshooting

### Error: "OpenAI API key not found"

**Solutions:**
* Check that `.env` file exists in the `optimized/` directory
* Verify the file contains: `OPENAI_API_KEY=sk-...`
* Ensure no extra spaces around the `=` sign
* Restart the Streamlit app after adding the key

### Error: "No locations were extracted"

**Possible causes:**
* LLM returned locations that couldn't be geocoded
* Location names were too vague (e.g., "a temple" instead of "Senso-ji Temple")
* Geocoding API temporarily unavailable

**Solutions:**
* Try regenerating the itinerary
* In draft stage, use more specific location names
* Check console output for geocoding errors
* Verify internet connection

### Map not displaying

**Solutions:**
* Check internet connection (needs to load OpenStreetMap tiles)
* Try a different browser (Chrome/Firefox recommended)
* Clear browser cache
* Check browser console for JavaScript errors
* Ensure port 8501 is not blocked by firewall

### App is very slow

**Possible causes:**
* OpenAI API experiencing high load
* Poor internet connection
* Many locations to geocode

**Solutions:**
* Be patient - AI generation takes time
* Check OpenAI status: https://status.openai.com/
* Reduce number of days in itinerary
* Simplify location requests

### "JSON parsing error" in draft or final stage

**Solutions:**
* This is a prompt engineering issue - try again
* If persistent, check agents/draft_agent.py or agents/final_agent.py
* The `strip_json()` helper should handle markdown code blocks
* File an issue if it keeps happening

### Downloaded itinerary has formatting issues

**Solutions:**
* Use a Markdown viewer (VS Code, Typora, or online viewer)
* Copy to a text editor that supports Markdown
* The file is plain text - formatting shows in Markdown renderers

## Tech Stack

* **LangGraph** - Multi-agent orchestration framework for building reliable LLM workflows
* **GPT-4o** - OpenAI's most advanced model for itinerary generation and location extraction
* **Streamlit** - Python web framework for rapid UI development with native Python
* **Folium** - Python library for interactive Leaflet.js maps with full customization
* **OpenStreetMap** - Free, community-driven map tiles and data
* **Nominatim** - Free geocoding API from OpenStreetMap (no API key required)
* **Python 3.8+** - Core programming language with async support
* **python-dotenv** - Secure environment variable management
* **Requests** - HTTP library for API calls

## Project Structure

```
optimized/
├── agents/
│   ├── draft_agent.py          # Draft generation workflow
│   └── final_agent.py          # Final itinerary + location extraction
├── maps/
│   └── map_utils.py            # Map creation and visualization
├── utils/
│   ├── llm.py                  # LLM factory function
│   ├── cleanup.py              # JSON cleaning helper
│   └── geo.py                  # Geocoding utilities
├── streamlit_app.py            # Main UI application
├── requirements.txt            # Python dependencies
└── .env                        # API keys (not committed)
```

## Example Usage

**Input:**
```
Destination: Kyoto, Japan
Duration: 5 days
Interests: temples, gardens, traditional culture, matcha
Budget: Medium
```

**Draft Output (Stage 2):**
```json
[
  {
    "day": 1,
    "main_destination": "Eastern Kyoto (Higashiyama)",
    "places": ["Kiyomizu-dera Temple", "Sannenzaka Street", "Yasaka Shrine"]
  },
  {
    "day": 2,
    "main_destination": "Arashiyama",
    "places": ["Bamboo Grove", "Tenryu-ji Temple", "Togetsukyo Bridge"]
  }
  ...
]
```

**User Edits:**
- Adds "Fushimi Inari Shrine" to Day 1
- Changes Day 3 to "Gion District"
- Adds in "Anything Else": "I want to try kaiseki cuisine"

**Final Output (Stage 3):**
- 5-page detailed itinerary with timings
- 6-8 locations plotted on map with routes
- Kaiseki restaurant recommendations included
- Fushimi Inari added to Day 1 with timing and description

## Limitations

* Routes shown are straight lines (not actual roads)
* Maximum 10 days supported for color coding (can extend in code)
* Requires internet for LLM calls, geocoding, and map tiles
* No offline mode
* No database - itineraries not saved between sessions
* Single-city trips only (multi-city requires manual editing)

## Future Enhancements

Planned features (contributions welcome):

- [ ] Real road routing using OpenRouteService API
- [ ] Database integration for saving itineraries
- [ ] User authentication and profile management
- [ ] Share itineraries via unique links
- [ ] Export to PDF with embedded map
- [ ] Multi-city trip support
- [ ] Weather information integration
- [ ] Flight and hotel search integration
- [ ] Mobile-responsive design improvements
- [ ] Offline mode with cached maps

## Credits

* **LangGraph** by LangChain - Enables the multi-agent architecture
* **OpenAI** - GPT-4o powers the intelligent itinerary generation
* **Streamlit** - Makes Python web apps incredibly easy to build
* **Folium** - Brings interactive maps to Python
* **OpenStreetMap** - Free, open-source map data
* **Nominatim** - Free geocoding service

---

Built with ❤️ for travelers who want AI-powered planning with complete control | [View Project Structure](FILE_STRUCTURE.md)