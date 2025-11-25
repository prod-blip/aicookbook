"""Map creation utilities using Folium"""
import folium
from folium import plugins
from typing import List, Dict


# Color palette for different days (supports up to 10 days)
DAY_COLORS = {
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


def create_itinerary_map(locations: List[Dict], center_location: str = "") -> folium.Map:
    """
    Creates an interactive map with itinerary locations and routes.
    
    Features:
    - Day-wise color coding
    - Numbered markers for each location
    - Routes connecting locations within each day
    - Layer control for filtering by day
    - Auto-fit bounds to show all locations
    
    Args:
        locations: List of location dictionaries with lat, lon, name, day
        center_location: Optional center location name (not used, kept for compatibility)
        
    Returns:
        Folium map object
    """
    if not locations:
        # Return default world map if no locations
        return folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")

    # Calculate center from all locations
    avg_lat = sum(loc['lat'] for loc in locations) / len(locations)
    avg_lon = sum(loc['lon'] for loc in locations) / len(locations)

    # Create base map
    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    # Group locations by day
    locations_by_day = {}
    for loc in locations:
        day = loc.get('day', 1)
        if day not in locations_by_day:
            locations_by_day[day] = []
        locations_by_day[day].append(loc)

    # Create layers for each day
    for day in sorted(locations_by_day.keys()):
        day_locs = locations_by_day[day]
        color = DAY_COLORS.get(day, 'gray')

        # Create feature group for this day (enables layer control)
        feature_group = folium.FeatureGroup(name=f'Day {day}', show=True)

        # Add markers for each location in this day
        for idx, loc in enumerate(day_locs, 1):
            # Main marker with popup
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=f"<b>Day {day} - Stop {idx}</b><br>{loc['name']}<br>{loc['display_name']}",
                tooltip=f"Day {day}: {loc['name']}",
                icon=folium.Icon(color=color, icon='info-sign', prefix='fa')
            ).add_to(feature_group)

            # Numbered circle marker overlay
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

            # Static polyline
            folium.PolyLine(
                coordinates,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f'Day {day} Route'
            ).add_to(feature_group)

            # Animated ant path for better visualization
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

    # Fit bounds to show all markers with padding
    if locations:
        bounds = [[loc['lat'], loc['lon']] for loc in locations]
        m.fit_bounds(bounds, padding=(30, 30))

    return m


def create_selection_map() -> folium.Map:
    """
    Creates a simple interactive map for location selection.
    
    Note: Currently not used in the optimized version, but kept for compatibility.
    
    Returns:
        Folium map object centered on world view
    """
    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles="OpenStreetMap"
    )
    
    # Add info marker
    folium.Marker(
        [0, 0],
        popup="Click anywhere on the map to select a location",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    
    m.add_child(folium.LatLngPopup())
    
    return m