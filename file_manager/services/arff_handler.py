# file_manager/services/arff_handler.py
"""
ARFF file handler implementation.
Single Responsibility: Handle ARFF (Attribute-Relation File Format) files.
ARFF is used by Weka and other machine learning tools.
"""
from typing import Dict, Any
from .base_handler import FileHandler


class ARFFHandler(FileHandler):
    """Handler for ARFF (Attribute-Relation File Format) files."""
    
    MAX_LINES_DISPLAY = 500  # Limit lines for performance
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read and parse ARFF file.
        
        Args:
            file_path: Path to ARFF file
        
        Returns:
            Dictionary with parsed ARFF data
        
        Raises:
            Exception: If file cannot be read or parsed
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception("No se pudo decodificar el archivo ARFF")
            
            # Parse ARFF structure
            lines = content.strip().split('\n')
            
            # Extract metadata
            relation_name = None
            attributes = []
            data_section = False
            data_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                line_lower = line_stripped.lower()
                
                # Skip empty lines and comments
                if not line_stripped or line_stripped.startswith('%'):
                    continue
                
                # Parse relation name
                if line_lower.startswith('@relation'):
                    relation_name = line_stripped.split(None, 1)[1] if len(line_stripped.split()) > 1 else 'Unknown'
                
                # Parse attributes
                elif line_lower.startswith('@attribute'):
                    attributes.append(line_stripped)
                
                # Data section marker
                elif line_lower.startswith('@data'):
                    data_section = True
                
                # Data lines
                elif data_section:
                    data_lines.append(line_stripped)
            
            return {
                'relation': relation_name,
                'attributes': attributes,
                'data_lines': data_lines,
                'raw_content': content,
            }
        
        except Exception as e:
            raise Exception(f"Error leyendo archivo ARFF: {str(e)}")
    
    def format_for_display(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format ARFF content for HTML display.
        
        Args:
            content: Parsed ARFF data
        
        Returns:
            Dictionary with formatted content and metadata
        """
        # Limit data lines for display
        data_lines = content['data_lines']
        total_data_lines = len(data_lines)
        display_data_lines = data_lines[:self.MAX_LINES_DISPLAY]
        
        # Format attributes for display
        formatted_attributes = []
        for attr in content['attributes']:
            formatted_attributes.append(attr.replace('@attribute', '').replace('@ATTRIBUTE', '').strip())
        
        return {
            'type': 'arff',
            'relation': content['relation'],
            'attributes': formatted_attributes,
            'attribute_count': len(formatted_attributes),
            'data_lines': display_data_lines,
            'total_data_lines': total_data_lines,
            'displayed_data_lines': len(display_data_lines),
            'is_truncated': total_data_lines > self.MAX_LINES_DISPLAY,
            'raw_content': content['raw_content'],
        }
