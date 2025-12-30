# file_manager/validators/file_validators.py
"""
File validation utilities.
Single Responsibility Principle: Only handles file validation.
"""
import os
import re
from django.core.exceptions import ValidationError
from django.conf import settings


class FileValidator:
    """
    Validator class for uploaded files.
    Validates extensions, size, and sanitizes filenames.
    """
    
    # We will use settings dynamically in methods, but keep these as fallback or context
    # BETTER: Just use settings directly to avoid duplication
    
    @staticmethod
    def validate_extension(filename: str, allowed_extensions: set = None) -> bool:
        if allowed_extensions is None:
            allowed_extensions = getattr(settings, 'ALLOWED_FILE_EXTENSIONS', ['csv', 'txt', 'json', 'html', 'arff'])
        
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        if ext not in allowed_extensions:
            raise ValidationError(
                f"Extension '{ext}' not allowed. Allowed: {', '.join(allowed_extensions)}"
            )
        
        return True
    
    @staticmethod
    def validate_size(file, max_size: int = None) -> bool:
        if max_size is None:
            max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 20 * 1024 * 1024)
        
        if hasattr(file, 'size') and file.size > max_size:
            max_mb = max_size / (1024 * 1024)
            raise ValidationError(
                f"File size ({file.size / (1024 * 1024):.2f} MB) exceeds "
                f"maximum allowed size ({max_mb:.0f} MB)"
            )
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Remove dangerous characters from filename.
        Prevents directory traversal and special character issues.
        
        Args:
            filename: Original filename
        
        Returns:
            Sanitized filename safe for storage
        """
        # Get base name to prevent directory traversal
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        # Keep only alphanumeric, dots, hyphens, and underscores
        filename = re.sub(r'[^\w\s.-]', '', filename)
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Remove multiple consecutive dots (security)
        filename = re.sub(r'\.{2,}', '.', filename)
        
        # Ensure filename is not empty
        if not filename or filename == '.':
            filename = 'unnamed_file.txt'
        
        return filename.lower()
    
    @staticmethod
    def validate_file(file) -> tuple[bool, str]:
        """
        Comprehensive file validation.
        
        Args:
            file: Django UploadedFile object
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate extension
            FileValidator.validate_extension(file.name)
            
            # Validate size
            FileValidator.validate_size(file)
            
            return True, ""
        
        except ValidationError as e:
            return False, str(e)
