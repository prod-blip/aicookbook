"""Geocoding utilities using Nominatim (OpenStreetMap)"""
import requests
from typing import Optional, Dict


def geocode(location_name: str) -> Optional[Dict[str, any]]:
    """
    Forward geocoding: Convert location name to coordinates.
    
    Args:
        location_name: Name of location (e.g., "Eiffel Tower, Paris")
        
    Returns:
        Dictionary with lat, lon, display_name or None if not found
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': location_name,
        'format': 'json',
        'limit': 1
    }
    headers = {'User-Agent': 'ItineraryPlanner/1.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if data:
            return {
                'lat': float(data[0]['lat']),
                'lon': float(data[0]['lon']),
                'display_name': data[0].get('display_name', location_name)
            }
        return None
        
    except Exception as e:
        print(f"❌ Geocoding error for {location_name}: {e}")
        return None


def reverse_geocode(lat: float, lon: float) -> str:
    """
    Reverse geocoding: Convert coordinates to location name.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Location name (country or city) or coordinates as string
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        'lat': lat,
        'lon': lon,
        'format': 'json'
    }
    headers = {'User-Agent': 'ItineraryPlanner/1.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        # Try to get country or city
        address = data.get('address', {})
        location = address.get('country', '') or address.get('city', '') or data.get('display_name', '')
        
        return location
        
    except Exception as e:
        print(f"❌ Reverse geocoding error: {e}")
        return f"{lat:.2f}, {lon:.2f}"