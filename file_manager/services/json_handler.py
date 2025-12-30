# file_manager/services/json_handler.py
"""
JSON file handler implementation.
Single Responsibility: Handle JSON files only.
"""
import json
from typing import Dict, Any
from .base_handler import FileHandler


class JSONHandler(FileHandler):
    """Handler for JSON files."""
    
    def read_file(self, file_path: str) -> Any:
        """
        Read and parse JSON file.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Parsed JSON data (dict, list, etc.)
        
        Raises:
            Exception: If file cannot be read or parsed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing JSON: {str(e)}")
        except Exception as e:
            raise Exception(f"Error leyendo archivo JSON: {str(e)}")
    
    def format_for_display(self, content: Any) -> Dict[str, Any]:
        """
        Format JSON content for HTML display.
        
        Args:
            content: Parsed JSON data
        
        Returns:
            Dictionary with formatted JSON and metadata
        """
        # Pretty print JSON with indentation
        formatted_json = json.dumps(content, indent=2, ensure_ascii=False)
        
        # Determine data type
        data_type = type(content).__name__
        
        # Count items if it's a collection
        item_count = None
        if isinstance(content, dict):
            item_count = len(content)
        elif isinstance(content, list):
            item_count = len(content)
        
        return {
            'type': 'json',
            'content': formatted_json,
            'data_type': data_type,
            'item_count': item_count,
            'is_dict': isinstance(content, dict),
            'is_list': isinstance(content, list),
        }
