# file_manager/services/txt_handler.py
"""
TXT file handler implementation.
Single Responsibility: Handle plain text files only.
"""
from typing import Dict, Any
from .base_handler import FileHandler


class TXTHandler(FileHandler):
    """Handler for plain text (.txt) files."""
    
    MAX_LINES_DISPLAY = 1000  # Limit lines for performance
    
    def read_file(self, file_path: str) -> str:
        """
        Read text file content.
        
        Args:
            file_path: Path to text file
        
        Returns:
            File content as string
        
        Raises:
            Exception: If file cannot be read
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    return content
                except UnicodeDecodeError:
                    continue
            
            raise Exception("No se pudo decodificar el archivo TXT con las codificaciones soportadas")
        
        except Exception as e:
            raise Exception(f"Error leyendo archivo TXT: {str(e)}")
    
    def format_for_display(self, content: str) -> Dict[str, Any]:
        """
        Format text content for HTML display.
        
        Args:
            content: Text file content
        
        Returns:
            Dictionary with formatted text and metadata
        """
        lines = content.splitlines()
        total_lines = len(lines)
        
        # Limit lines for display
        display_lines = lines[:self.MAX_LINES_DISPLAY]
        display_content = '\n'.join(display_lines)
        
        return {
            'type': 'txt',
            'content': display_content,
            'lines_total': total_lines,
            'lines_displayed': len(display_lines),
            'is_truncated': total_lines > self.MAX_LINES_DISPLAY,
            'characters': len(content),
            'preview_mode': total_lines > self.MAX_LINES_DISPLAY,
        }
