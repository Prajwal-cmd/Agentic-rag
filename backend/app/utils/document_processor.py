"""
Document Processing Utilities
Pattern: Recursive chunking for optimal context preservation
Source: LangChain text splitters
"""
from typing import List, Dict
import PyPDF2
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessor:
    """
    Process various document formats into chunks for embedding.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter - industry standard
        # Source: LangChain documentation
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Hierarchical splitting
            length_function=len
        )
    
    def extract_text_from_pdf(self, file_bytes: BytesIO) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file_bytes)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ValueError(f"Failed to extract PDF text: {e}")
    
    def extract_text_from_docx(self, file_bytes: BytesIO) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_bytes)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise ValueError(f"Failed to extract DOCX text: {e}")
    
    def extract_text_from_txt(self, file_bytes: bytes) -> str:
        """Extract text from TXT file"""
        try:
            text = file_bytes.decode('utf-8')
            return text.strip()
        except UnicodeDecodeError:
            # Try alternative encodings
            try:
                text = file_bytes.decode('latin-1')
                return text.strip()
            except Exception as e:
                logger.error(f"TXT decoding error: {e}")
                raise ValueError(f"Failed to decode text file: {e}")
    
    def process_file(self, file_bytes: BytesIO, filename: str) -> List[Dict]:
        """
        Process file into chunks with metadata.
        
        Args:
            file_bytes: Raw file bytes as BytesIO
            filename: Original filename
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Extract text based on file type
        extension = filename.lower().split('.')[-1]
        
        if extension == 'pdf':
            text = self.extract_text_from_pdf(file_bytes)
        elif extension in ['docx', 'doc']:
            text = self.extract_text_from_docx(file_bytes)
        elif extension == 'txt':
            # For txt, we need raw bytes
            file_bytes.seek(0)
            text = self.extract_text_from_txt(file_bytes.read())
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        logger.info(f"Extracted {len(text)} characters from {filename}")
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        
        # Create chunk dictionaries with metadata
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dicts.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return chunk_dicts

# Global processor instance
document_processor = None

def get_document_processor(chunk_size: int = 1000, chunk_overlap: int = 200) -> DocumentProcessor:
    """Get or create global document processor instance"""
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor(chunk_size, chunk_overlap)
    return document_processor