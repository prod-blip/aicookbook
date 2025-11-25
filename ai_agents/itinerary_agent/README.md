# AI Itinerary Planner

An intelligent travel planning assistant that generates personalized day-by-day itineraries with interactive route maps, featuring a human-in-the-loop workflow for complete user control over destinations and activities.

✨ **Powered by LangGraph multi-agent architecture with GPT-4o and interactive Folium maps**



https://github.com/user-attachments/assets/9e647dab-cd78-44ba-88ea-69c5f48d066b



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

1. Clone this repository:

```bash
git clone https://github.com/prod-blip/aicookbook.git
cd aicookbook/ai_agents/itinerary_agent
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

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

1. Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

2. Your browser will automatically open to `http://localhost:8501`

3. If the browser doesn't open automatically, manually navigate to the URL shown in the terminal

4. You'll see the three-stage workflow interface ready to use

## How It Works - Complete Workflow

### Stage 1: Input Trip Details

### Stage 2: Review & Edit Draft

### Stage 3: View Final Results

**What you get:**

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
