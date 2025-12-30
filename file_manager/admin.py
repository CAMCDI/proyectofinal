# file_manager/admin.py
"""Admin configuration for file_manager app."""
from django.contrib import admin
from .models import UploadedFile


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    """Admin interface for UploadedFile model."""
    
    list_display = ['filename', 'file_type', 'file_size_display', 'uploaded_at']
    list_filter = ['file_type', 'uploaded_at']
    search_fields = ['filename']
    readonly_fields = ['uploaded_at', 'file_size']
    date_hierarchy = 'uploaded_at'
    
    def file_size_display(self, obj):
        """Display file size in human-readable format."""
        return obj.get_file_size_display()
    
    file_size_display.short_description = 'Tamaño'
