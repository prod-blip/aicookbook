"""Text cleanup utilities"""


def strip_json(text: str) -> str:
    """
    Removes markdown code block formatting from JSON strings.
    
    Args:
        text: Raw text that may contain ```json markers
        
    Returns:
        Cleaned JSON string
    """
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "")
    return text.strip()