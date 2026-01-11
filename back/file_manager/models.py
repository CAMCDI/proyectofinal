# file_manager/models.py
"""
Models for file upload management.
Following Clean Code principles: Single Responsibility, Clear Naming.
"""
from django.db import models
from django.core.validators import FileExtensionValidator
import os


class UploadedFile(models.Model):
    """
    Model to track and manage uploaded files.
    
    Attributes:
        file: The actual file stored in media/uploads
        filename: Original filename for display
        file_type: Extension/type of file (csv, txt, json, html)
        uploaded_at: Timestamp of upload
        file_size: Size in bytes
    """
    
    ALLOWED_EXTENSIONS = ['csv', 'txt', 'json', 'html', 'arff', 'zip', '']
    
    file = models.FileField(
        upload_to='uploads/%Y/%m/%d/',
        validators=[FileExtensionValidator(allowed_extensions=ALLOWED_EXTENSIONS)],
        help_text="Supported formats: CSV, TXT, JSON, HTML"
    )
    filename = models.CharField(
        max_length=255,
        help_text="Original filename"
    )
    file_type = models.CharField(
        max_length=10,
        choices=[(ext, ext.upper() if ext else 'SIN EXTENSIÓN') for ext in ALLOWED_EXTENSIONS],
        help_text="File extension"
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Upload timestamp"
    )
    file_size = models.IntegerField(
        help_text="File size in bytes"
    )

    # Status fields for Async Processing
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    analysis_status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )
    analysis_result = models.JSONField(
        null=True, 
        blank=True,
        help_text="Stored ML analysis result"
    )
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = "Uploaded File"
        verbose_name_plural = "Uploaded Files"
    
    def __str__(self):
        """String representation of the model."""
        return f"{self.filename} ({self.file_type}) - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_file_size_display(self):
        """Return human-readable file size."""
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    
    def save(self, *args, **kwargs):
        """Override save to automatically set filename and file_type."""
        if self.file:
            self.filename = os.path.basename(self.file.name)
            if '.' in self.filename:
                self.file_type = self.filename.split('.')[-1].lower()
            else:
                self.file_type = '' # Empty extension
            
            if hasattr(self.file, 'size'):
                self.file_size = self.file.size
        super().save(*args, **kwargs)
