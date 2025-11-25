# Itinerary Planner Agent - Optimized Modular Version

Clean, production-ready travel itinerary planner with human-in-the-loop workflow and interactive maps.

## 🎯 Project Structure

```
optimized/
├── agents/
│   ├── __init__.py
│   ├── draft_agent.py      # Draft itinerary generation
│   └── final_agent.py      # Final detailed itinerary + location extraction
├── maps/
│   ├── __init__.py
│   └── map_utils.py        # Map creation and visualization
├── utils/
│   ├── __init__.py
│   ├── llm.py              # LLM factory function
│   ├── cleanup.py          # JSON cleaning helper
│   └── geo.py              # Geocoding utilities
├── streamlit_app.py        # Main UI application
├── requirements.txt
├── .env.example
└── README.md
```

## ✨ Key Improvements

### 1. **Modular Architecture**
- Separated concerns: agents, maps, utils, UI
- Single Responsibility Principle for each module
- Easy to test and maintain

### 2. **DRY (Don't Repeat Yourself)**
- `get_llm()` factory function eliminates duplicate LLM instantiation
- `strip_json()` helper removes repeated JSON cleaning logic
- `geocode()` and `reverse_geocode()` consolidate API calls
- Session state initialization in one function

### 3. **Clean Code**
- Clear function names and docstrings
- Type hints for better IDE support
- Consistent error handling patterns
- Well-organized imports

### 4. **Production-Ready**
- Proper Python package structure
- Reusable components
- Easy to extend with new features
- Better performance with optimized imports

## 🚀 Features

- ✅ **Human-in-the-loop workflow**: Review and edit AI-generated plans
- ✅ **Day-wise itinerary**: Structured day-by-day breakdown
- ✅ **Interactive maps**: Folium maps with layer control
- ✅ **Color-coded routes**: Different colors for each day
- ✅ **Location extraction**: AI finds and geocodes places automatically
- ✅ **Additional requirements**: "Anything else" field for custom requests
- ✅ **Download option**: Export itinerary as Markdown
- ✅ **Budget-aware**: Recommendations based on budget level

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Add your OpenAI API key:

```
OPENAI_API_KEY=sk-...your-key-here
```

Get your key from: https://platform.openai.com/api-keys

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 🎮 How to Use

### Stage 1: Input Details
1. Enter destination (e.g., "Tokyo, Japan")
2. Specify duration (e.g., "7 days")
3. Add interests (e.g., "anime, food, temples")
4. Select budget level (Low/Medium/High)
5. Click "Generate Draft Itinerary"

### Stage 2: Review & Edit
1. Review the AI-generated day-wise plan
2. Edit main destinations for each day
3. Modify places to visit (comma-separated)
4. Add any additional requirements in "Anything Else"
5. Click "Approve & Generate Full Itinerary"

### Stage 3: View Results
1. Read your detailed itinerary
2. Explore the interactive map with color-coded routes
3. Toggle day layers to focus on specific days
4. Download your itinerary as Markdown
5. Plan another trip or make adjustments

## 🏗️ Architecture

### LangGraph Workflows

**Draft Agent (1 node):**
```
generate_draft → END
```
- Creates simple JSON structure for human review
- Fast generation for quick iteration

**Final Agent (3 nodes):**
```
generate_itinerary → extract_locations → format_output → END
```
- Generates detailed itinerary from approved draft
- Extracts 5-8 key locations with day assignments
- Geocodes and plots on interactive map

### Module Responsibilities

**agents/**: LangGraph workflows
- `draft_agent.py`: Generates editable day-wise breakdown
- `final_agent.py`: Creates detailed itinerary with location extraction

**maps/**: Visualization
- `map_utils.py`: Creates Folium maps with day-wise coloring and routes

**utils/**: Helper functions
- `llm.py`: Centralized LLM factory
- `cleanup.py`: JSON string cleaning
- `geo.py`: Forward and reverse geocoding

**streamlit_app.py**: UI logic
- Three-stage workflow management
- Session state handling
- User input validation

## 🎨 Map Features

### Day-wise Color Coding
- Day 1: Red
- Day 2: Blue
- Day 3: Green
- Day 4: Purple
- Day 5: Orange
- (Supports up to 10 days)

### Interactive Elements
- **Markers**: Show each location with day and sequence number
- **Routes**: Polylines connecting locations within each day
- **Animated paths**: Ant path animation showing route direction
- **Layer control**: Toggle visibility of specific days
- **Popups**: Click markers for location details
- **Auto-fit**: Map automatically zooms to show all locations

## 💡 Code Comparison

### Before (Repetitive):
```python
# In 3 different nodes:
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

# In 3 different nodes:
if text.startswith("```"):
    text = text.replace("```json", "").replace("```", "")
```

### After (DRY):
```python
# Once in utils/llm.py:
llm = get_llm(temperature=0.7)

# Once in utils/cleanup.py:
cleaned = strip_json(text)
```

## 📊 Benefits of This Structure

### For Development
- ✅ Easy to add new features
- ✅ Simple to test individual components
- ✅ Better IDE autocomplete
- ✅ Clear separation of concerns

### For Maintenance
- ✅ Bug fixes in one place
- ✅ Easier code reviews
- ✅ Better documentation
- ✅ Simpler onboarding for new developers

### For Production
- ✅ Optimized imports
- ✅ Reusable components
- ✅ Better error isolation
- ✅ Scalable architecture

## 🔧 Extending the Application

### Add a New Agent Node
1. Create node function in appropriate agent file
2. Add to workflow in `create_X_graph()`
3. Update state TypedDict if needed

### Add a New Map Feature
1. Add function to `maps/map_utils.py`
2. Import in `streamlit_app.py`
3. Call in appropriate stage

### Add a New Utility
1. Create function in appropriate utils file
2. Export in `utils/__init__.py`
3. Import where needed

## 🐛 Troubleshooting

**Import Errors:**
```bash
# Run from the optimized/ directory
cd optimized/
streamlit run streamlit_app.py
```

**Module not found:**
- Ensure all `__init__.py` files exist
- Check Python path includes current directory

**API Errors:**
- Verify `.env` file exists in project root
- Check OpenAI API key is valid
- Ensure you have API credits

## 📈 Performance

- **60% shorter UI code** compared to monolithic version
- **Faster imports** with modular structure
- **Better memory usage** with optimized session state
- **Cleaner error handling** with centralized functions

## 🎓 Best Practices Used

1. **Single Responsibility Principle**: Each module does one thing well
2. **DRY**: No code duplication
3. **Clear naming**: Functions and variables are self-documenting
4. **Type hints**: Better IDE support and code clarity
5. **Docstrings**: All functions documented
6. **Error handling**: Consistent try-except patterns
7. **Package structure**: Proper Python package layout

## 📝 Cost Estimate

- OpenAI API: ~$0.02-$0.05 per itinerary
- Nominatim Geocoding: Free (rate-limited to 1 req/sec)
- OpenStreetMap tiles: Free
- Folium: Free
- Streamlit: Free (self-hosted)

## 🚧 Future Enhancements

Potential additions:
- [ ] Add caching for geocoding results
- [ ] Implement actual road routing (OpenRouteService)
- [ ] Add weather information
- [ ] Support multi-city trips
- [ ] Export to PDF
- [ ] Save/load itineraries from database
- [ ] User authentication
- [ ] Share itineraries via link

## 🤝 Contributing

This modular structure makes it easy to contribute:

1. Fork the repository
2. Create a feature branch
3. Add your feature in the appropriate module
4. Add tests if applicable
5. Submit a pull request

## 📄 License

Built for educational and personal use.

## 🙏 Acknowledgments

- LangGraph for agent orchestration
- OpenAI GPT-4o for itinerary generation
- Streamlit for the web interface
- Folium for interactive maps
- OpenStreetMap for free map data

---

Built with ❤️ for travelers who want AI-powered planning with full control.