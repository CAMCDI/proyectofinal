# file_manager/forms.py
"""
Forms for file upload.
Clean separation of validation concerns.
"""
from django import forms
from django.conf import settings
from .models import UploadedFile
from .validators import FileValidator


class FileUploadForm(forms.ModelForm):
    """
    Form for uploading files.
    Includes client-side and server-side validation.
    """
    
    class Meta:
        model = UploadedFile
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'id': 'fileInput',
                'accept': ','.join([f'.{ext}' for ext in settings.ALLOWED_FILE_EXTENSIONS]),
            })
        }
        labels = {
            'file': 'Selecciona un archivo'
        }
        help_texts = {
            'file': f'Formatos permitidos: {", ".join([ext.upper() for ext in settings.ALLOWED_FILE_EXTENSIONS])}. Tamaño máximo: {settings.MAX_UPLOAD_SIZE // (1024*1024)} MB'
        }
    
    def clean_file(self):
        """
        Validate uploaded file.
        Uses FileValidator for comprehensive validation.
        """
        file = self.cleaned_data.get('file')
        
        if file:
            # Validate using FileValidator
            is_valid, error_message = FileValidator.validate_file(file)
            
            if not is_valid:
                raise forms.ValidationError(error_message)
            
            # Sanitize filename
            file.name = FileValidator.sanitize_filename(file.name)
        
        return file
