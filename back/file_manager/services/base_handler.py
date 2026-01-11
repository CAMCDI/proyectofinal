# file_manager/services/base_handler.py
"""
Base handler interface for file processing.
Follows Open/Closed Principle - open for extension, closed for modification.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class FileHandler(ABC):
    """
    Abstract base class for file handlers.
    Defines the interface that all file handlers must implement.
    """
    
    @abstractmethod
    def read_file(self, file_path: str) -> Any:
        """
        Read and parse the file content.
        
        Args:
            file_path: Path to the file to read
        
        Returns:
            Parsed file content in appropriate format
        """
        pass
    
    @abstractmethod
    def format_for_display(self, content: Any) -> Dict[str, Any]:
        """
        Format parsed content for HTML display.
        
        Args:
            content: Parsed file content
        
        Returns:
            Dictionary with formatted content ready for template rendering
        """
        pass
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Complete file processing workflow.
        Template method pattern - defines the algorithm structure.
        
        Args:
            file_path: Path to the file to process
        
        Returns:
            Formatted content ready for display
        """
        content = self.read_file(file_path)
        return self.format_for_display(content)
