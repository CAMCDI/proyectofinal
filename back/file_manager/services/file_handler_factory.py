# file_manager/services/file_handler_factory.py
"""
Factory for creating appropriate file handlers.
Follows Factory Pattern and Open/Closed Principle.
"""
from typing import Optional
from .base_handler import FileHandler
from .csv_handler import CSVHandler
from .txt_handler import TXTHandler
from .json_handler import JSONHandler
from .html_handler import HTMLHandler
from .arff_handler import ARFFHandler


class FileHandlerFactory:
    """
    Factory class to create appropriate file handlers based on file extension.
    Open/Closed Principle: Can add new handlers without modifying this class structure.
    """
    
    # Registry of handlers (Strategy pattern)
    _handlers = {
        'csv': CSVHandler,
        'txt': TXTHandler,
        'json': JSONHandler,
        'html': HTMLHandler,
        'arff': ARFFHandler,
    }
    
    @classmethod
    def get_handler(cls, file_extension: str) -> Optional[FileHandler]:
        """
        Get appropriate handler for file extension.
        
        Args:
            file_extension: File extension (without dot)
        
        Returns:
            FileHandler instance or None if extension not supported
        """
        extension = file_extension.lower().strip('.')
        handler_class = cls._handlers.get(extension)
        
        if handler_class:
            return handler_class()
        
        return None
    
    @classmethod
    def register_handler(cls, extension: str, handler_class: type):
        """
        Register a new handler for an extension.
        Allows extending functionality without modifying existing code.
        
        Args:
            extension: File extension to handle
            handler_class: Handler class (must inherit from FileHandler)
        """
        if not issubclass(handler_class, FileHandler):
            raise TypeError("Handler class must inherit from FileHandler")
        
        cls._handlers[extension.lower().strip('.')] = handler_class
    
    @classmethod
    def get_supported_extensions(cls) -> list:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return list(cls._handlers.keys())
