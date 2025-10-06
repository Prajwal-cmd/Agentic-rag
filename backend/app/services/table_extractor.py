"""
Table Extraction Service from PDFs

Pattern: Multi-library fallback (Camelot → pdfplumber → Tabula)
Source: Unstract.io (2025), Table extraction best practices

Features:
- Extract tables from PDFs
- Export to CSV/Excel
- Handle complex table layouts
- OCR support for scanned PDFs
"""

from typing import List, Dict, Any, Optional
import io
import tempfile
import os
from pathlib import Path

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class TableExtractor:
    """
    Multi-library PDF table extraction with fallback chain.
    
    Pattern: Camelot (best quality) → pdfplumber (fallback)
    Source: Unstract.io table extraction benchmarks (2025)
    """
    
    def __init__(self):
        self.supported_methods = []
        
        if CAMELOT_AVAILABLE:
            self.supported_methods.append("camelot")
            logger.info("✓ Camelot table extraction available (primary)")
        
        if PDFPLUMBER_AVAILABLE:
            self.supported_methods.append("pdfplumber")
            logger.info("✓ pdfplumber table extraction available (fallback)")
        
        if not self.supported_methods:
            logger.error("❌ No table extraction libraries available!")
    
    def extract_tables_from_pdf(
        self,
        pdf_path: str,
        pages: str = "all",
        flavor: str = "lattice"
    ) -> List[Dict[str, Any]]:
        """
        Extract all tables from PDF with metadata.
        
        Args:
            pdf_path: Path to PDF file
            pages: Pages to extract ("all", "1-3", "1,3,5")
            flavor: Extraction method ("lattice" for bordered tables, 
                   "stream" for borderless)
        
        Returns:
            List of dicts with table data and metadata:
            [
                {
                    "page": 1,
                    "table_number": 1,
                    "data": pandas.DataFrame,
                    "bbox": [x1, y1, x2, y2],
                    "extraction_method": "camelot"
                }
            ]
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return []
        
        # ATTEMPT 1: Camelot (best for complex tables)
        if "camelot" in self.supported_methods:
            try:
                logger.info(f"🔍 ATTEMPT 1: Camelot extraction from {pdf_path}")
                tables = self._extract_with_camelot(pdf_path, pages, flavor)
                if tables:
                    logger.info(f"✅ Camelot extracted {len(tables)} tables")
                    return tables
                logger.warning("Camelot returned 0 tables, trying pdfplumber...")
            except Exception as e:
                logger.warning(f"⚠️ Camelot failed: {e}, falling back to pdfplumber")
        
        # ATTEMPT 2: pdfplumber (fallback)
        if "pdfplumber" in self.supported_methods:
            try:
                logger.info(f"🔍 ATTEMPT 2: pdfplumber extraction from {pdf_path}")
                tables = self._extract_with_pdfplumber(pdf_path, pages)
                if tables:
                    logger.info(f"✅ pdfplumber extracted {len(tables)} tables")
                    return tables
                logger.warning("pdfplumber returned 0 tables")
            except Exception as e:
                logger.error(f"⚠️ pdfplumber failed: {e}")
        
        logger.error("❌ All table extraction methods failed")
        return []
    
    def _extract_with_camelot(
        self,
        pdf_path: str,
        pages: str,
        flavor: str
    ) -> List[Dict[str, Any]]:
        """
        Extract tables using Camelot (best quality).
        
        Camelot is best for:
        - Tables with clear borders (lattice)
        - Complex multi-column layouts
        - High-quality PDFs
        """
        try:
            # Extract tables
            tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor=flavor,
                suppress_stdout=True
            )
            
            if not tables:
                return []
            
            # Convert to standard format
            extracted = []
            for i, table in enumerate(tables, 1):
                extracted.append({
                    "page": table.page,
                    "table_number": i,
                    "data": table.df,
                    "bbox": table._bbox if hasattr(table, '_bbox') else None,
                    "extraction_method": "camelot",
                    "accuracy": table.parsing_report.get("accuracy", 0.0)
                })
            
            return extracted
        
        except Exception as e:
            logger.error(f"Camelot extraction error: {e}")
            raise
    
    def _extract_with_pdfplumber(
        self,
        pdf_path: str,
        pages: str
    ) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber (fallback).
        
        pdfplumber is best for:
        - Borderless tables
        - Simple layouts
        - When Camelot fails
        """
        try:
            extracted = []
            
            with pdfplumber.open(pdf_path) as pdf:
                # Parse page specification
                if pages == "all":
                    page_nums = range(len(pdf.pages))
                else:
                    page_nums = self._parse_page_spec(pages, len(pdf.pages))
                
                table_counter = 1
                for page_num in page_nums:
                    page = pdf.pages[page_num]
                    tables = page.extract_tables()
                    
                    for table_data in tables:
                        if not table_data or len(table_data) < 2:
                            continue
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        
                        extracted.append({
                            "page": page_num + 1,
                            "table_number": table_counter,
                            "data": df,
                            "bbox": None,
                            "extraction_method": "pdfplumber",
                            "accuracy": None
                        })
                        
                        table_counter += 1
            
            return extracted
        
        except Exception as e:
            logger.error(f"pdfplumber extraction error: {e}")
            raise
    
    def _parse_page_spec(self, pages: str, total_pages: int) -> List[int]:
        """Parse page specification string."""
        if pages == "all":
            return list(range(total_pages))
        
        page_nums = []
        for part in pages.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                page_nums.extend(range(start - 1, min(end, total_pages)))
            else:
                page_num = int(part) - 1
                if 0 <= page_num < total_pages:
                    page_nums.append(page_num)
        
        return sorted(set(page_nums))
    
    def export_tables(
        self,
        tables: List[Dict[str, Any]],
        output_format: str = "csv"
    ) -> List[Dict[str, Any]]:
        """
        Export extracted tables to CSV or Excel.
        
        Args:
            tables: List of extracted tables
            output_format: "csv" or "excel"
        
        Returns:
            List of dicts with file paths and metadata
        """
        exported = []
        
        for table in tables:
            try:
                df = table["data"]
                
                # Generate filename
                filename = f"table_p{table['page']}_n{table['table_number']}.{output_format}"
                
                # Create temp file
                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    suffix=f".{output_format}",
                    delete=False
                ) as tmp:
                    if output_format == "csv":
                        df.to_csv(tmp.name, index=False, encoding='utf-8')
                    elif output_format == "excel":
                        df.to_excel(tmp.name, index=False, engine='openpyxl')
                    
                    exported.append({
                        "filename": filename,
                        "filepath": tmp.name,
                        "page": table["page"],
                        "table_number": table["table_number"],
                        "rows": len(df),
                        "columns": len(df.columns),
                        "format": output_format
                    })
            
            except Exception as e:
                logger.error(f"Export failed for table {table['table_number']}: {e}")
        
        return exported
    
    def tables_to_markdown(self, tables: List[Dict[str, Any]]) -> str:
        """
        Convert extracted tables to Markdown format.
        
        Useful for LLM consumption and display.
        """
        markdown = ""
        
        for table in tables:
            df = table["data"]
            
            markdown += f"\n### Table {table['table_number']} (Page {table['page']})\n\n"
            markdown += df.to_markdown(index=False)
            markdown += "\n\n"
        
        return markdown
    
    def get_table_summary(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics of extracted tables."""
        if not tables:
            return {"total_tables": 0}
        
        return {
            "total_tables": len(tables),
            "pages_with_tables": len(set(t["page"] for t in tables)),
            "methods_used": list(set(t["extraction_method"] for t in tables)),
            "total_rows": sum(len(t["data"]) for t in tables),
            "total_columns": sum(len(t["data"].columns) for t in tables),
            "tables_by_page": {
                page: len([t for t in tables if t["page"] == page])
                for page in sorted(set(t["page"] for t in tables))
            }
        }

# Global instance
_table_extractor = None

def get_table_extractor() -> TableExtractor:
    """Get or create table extractor singleton."""
    global _table_extractor
    if _table_extractor is None:
        _table_extractor = TableExtractor()
    return _table_extractor
