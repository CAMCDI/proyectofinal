# file_manager/services/csv_handler.py
"""
CSV file handler implementation.
Single Responsibility: Handle CSV files only.
"""
import pandas as pd
from typing import Dict, Any
from .base_handler import FileHandler


class CSVHandler(FileHandler):
    """Handler for CSV (Comma-Separated Values) files."""
    
    MAX_ROWS_DISPLAY = 100  # Limit rows for performance
    
    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        Read CSV file using pandas.
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            pandas DataFrame with CSV data
        
        Raises:
            Exception: If file cannot be read or parsed
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail
            raise Exception("No se pudo decodificar el archivo CSV con las codificaciones soportadas")
        
        except Exception as e:
            raise Exception(f"Error leyendo archivo CSV: {str(e)}")
    
    def format_for_display(self, content: pd.DataFrame) -> Dict[str, Any]:
        """
        Format DataFrame for HTML display.
        
        Args:
            content: pandas DataFrame
        
        Returns:
            Dictionary with HTML table and metadata
        """
        # Limit rows for display
        display_df = content.head(self.MAX_ROWS_DISPLAY)
        
        # Convert to HTML table
        html_table = display_df.to_html(
            classes='table table-striped table-hover table-sm',
            index=True,
            border=0,
            justify='left'
        )
        
        return {
            'type': 'csv',
            'html_content': html_table,
            'rows_total': len(content),
            'rows_displayed': len(display_df),
            'columns': list(content.columns),
            'column_count': len(content.columns),
            'is_truncated': len(content) > self.MAX_ROWS_DISPLAY,
            'preview_mode': len(content) > self.MAX_ROWS_DISPLAY,
        }
