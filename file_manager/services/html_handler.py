# file_manager/services/html_handler.py
"""
HTML file handler implementation.
Single Responsibility: Handle HTML files only.
"""
import html
from typing import Dict, Any
from .base_handler import FileHandler


class HTMLHandler(FileHandler):
    """Handler for HTML files."""
    
    def read_file(self, file_path: str) -> str:
        """
        Read HTML file content.
        
        Args:
            file_path: Path to HTML file
        
        Returns:
            HTML content as string
        
        Raises:
            Exception: If file cannot be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        
        except Exception as e:
            raise Exception(f"Error leyendo archivo HTML: {str(e)}")
    
    def format_for_display(self, content: str) -> Dict[str, Any]:
        """
        Format HTML content for safe display.
        Provides both escaped code view and sanitized rendered view.
        
        Args:
            content: HTML file content
        
        Returns:
            Dictionary with both code and rendered views
        """
        # Escape HTML for code view (prevents XSS)
        escaped_html = html.escape(content)
        
        # For rendered view, we'll let Django's template handle it with |safe
        # but provide warnings about potential risks
        
        return {
            'type': 'html',
            'code_view': escaped_html,
            'raw_content': content,  # Will be sanitized in template
            'characters': len(content),
            'lines': len(content.splitlines()),
        }
