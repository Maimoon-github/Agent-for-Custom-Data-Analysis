"""
Document processing pipeline for the RAG Agent system.
Handles loading, processing, and chunking of various document formats.
"""

import logging
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Document loaders
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader
)

# Text splitters
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

from .config import DocumentProcessingConfig

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document representation."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    source: Optional[str] = None

@dataclass
class ProcessingResult:
    """Result of document processing."""
    success: bool
    documents: List[Document]
    error: Optional[str] = None
    processing_time: Optional[float] = None

class DocumentProcessor:
    """Document processing pipeline."""
    
    def __init__(self, config: DocumentProcessingConfig):
        """Initialize document processor.
        
        Args:
            config: Document processing configuration
        """
        self.config = config
        self.text_splitter = self._initialize_text_splitter()
        self.supported_loaders = self._initialize_loaders()
    
    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialize the text splitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def _initialize_loaders(self) -> Dict[str, Any]:
        """Initialize document loaders for supported formats."""
        return {
            'txt': TextLoader,
            'md': TextLoader,
            'pdf': PyPDFLoader,
            'csv': CSVLoader,
            'docx': UnstructuredWordDocumentLoader
        }
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a single file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            ProcessingResult containing processed documents or error
        """
        import time
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            
            # Validate file
            if not file_path.exists():
                return ProcessingResult(
                    success=False,
                    documents=[],
                    error=f"File not found: {file_path}"
                )
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                return ProcessingResult(
                    success=False,
                    documents=[],
                    error=f"File too large: {file_size_mb:.2f}MB > {self.config.max_file_size_mb}MB"
                )
            
            # Determine file type
            file_extension = file_path.suffix.lower().lstrip('.')
            if file_extension not in self.config.supported_formats:
                return ProcessingResult(
                    success=False,
                    documents=[],
                    error=f"Unsupported file format: {file_extension}"
                )
            
            # Load document
            loader_class = self.supported_loaders.get(file_extension)
            if not loader_class:
                return ProcessingResult(
                    success=False,
                    documents=[],
                    error=f"No loader available for format: {file_extension}"
                )
            
            # Load and process
            loader = loader_class(str(file_path))
            raw_documents = loader.load()
            
            # Split into chunks
            split_documents = self.text_splitter.split_documents(raw_documents)
            
            # Convert to our Document format
            processed_documents = []
            for i, doc in enumerate(split_documents):
                # Generate document ID
                doc_id = self._generate_doc_id(str(file_path), i)
                
                # Create metadata
                metadata = {
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_type': file_extension,
                    'chunk_index': i,
                    'total_chunks': len(split_documents),
                    'file_size_mb': file_size_mb,
                    'processing_timestamp': time.time()
                }
                
                # Add original metadata if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata.update(doc.metadata)
                
                processed_doc = Document(
                    content=doc.page_content,
                    metadata=metadata,
                    doc_id=doc_id,
                    source=str(file_path)
                )
                processed_documents.append(processed_doc)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Processed {file_path}: {len(processed_documents)} chunks in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                documents=processed_documents,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing {file_path}: {e}")
            return ProcessingResult(
                success=False,
                documents=[],
                error=str(e),
                processing_time=processing_time
            )
    
    def process_directory(self, 
                         directory_path: Union[str, Path], 
                         recursive: bool = True) -> List[ProcessingResult]:
        """Process all files in a directory.
        
        Args:
            directory_path: Path to directory to process
            recursive: Whether to process subdirectories
            
        Returns:
            List of ProcessingResults for each file
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found or not a directory: {directory_path}")
            return []
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        all_files = []
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file():
                file_extension = file_path.suffix.lower().lstrip('.')
                if file_extension in self.config.supported_formats:
                    all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} supported files in {directory_path}")
        
        # Process files in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in all_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results.append(ProcessingResult(
                        success=False,
                        documents=[],
                        error=str(e)
                    ))
        
        return results
    
    def process_text(self, 
                    text: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process raw text content.
        
        Args:
            text: Text content to process
            metadata: Optional metadata for the text
            
        Returns:
            ProcessingResult containing processed documents
        """
        import time
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                return ProcessingResult(
                    success=False,
                    documents=[],
                    error="Empty text provided"
                )
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create documents
            processed_documents = []
            for i, chunk in enumerate(chunks):
                # Generate document ID
                doc_id = self._generate_doc_id(text[:100], i)  # Use first 100 chars for ID
                
                # Create metadata
                chunk_metadata = {
                    'source': 'text_input',
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'processing_timestamp': time.time()
                }
                
                if metadata:
                    chunk_metadata.update(metadata)
                
                processed_doc = Document(
                    content=chunk,
                    metadata=chunk_metadata,
                    doc_id=doc_id,
                    source='text_input'
                )
                processed_documents.append(processed_doc)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                documents=processed_documents,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing text: {e}")
            return ProcessingResult(
                success=False,
                documents=[],
                error=str(e),
                processing_time=processing_time
            )
    
    def process_web_url(self, url: str) -> ProcessingResult:
        """Process content from a web URL.
        
        Args:
            url: URL to process
            
        Returns:
            ProcessingResult containing processed documents
        """
        import time
        start_time = time.time()
        
        try:
            # Load web content
            loader = WebBaseLoader([url])
            raw_documents = loader.load()
            
            if not raw_documents:
                return ProcessingResult(
                    success=False,
                    documents=[],
                    error=f"No content loaded from URL: {url}"
                )
            
            # Split into chunks
            split_documents = self.text_splitter.split_documents(raw_documents)
            
            # Convert to our Document format
            processed_documents = []
            for i, doc in enumerate(split_documents):
                # Generate document ID
                doc_id = self._generate_doc_id(url, i)
                
                # Create metadata
                metadata = {
                    'source': url,
                    'source_type': 'web',
                    'chunk_index': i,
                    'total_chunks': len(split_documents),
                    'processing_timestamp': time.time()
                }
                
                # Add original metadata if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata.update(doc.metadata)
                
                processed_doc = Document(
                    content=doc.page_content,
                    metadata=metadata,
                    doc_id=doc_id,
                    source=url
                )
                processed_documents.append(processed_doc)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Processed URL {url}: {len(processed_documents)} chunks in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                documents=processed_documents,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing URL {url}: {e}")
            return ProcessingResult(
                success=False,
                documents=[],
                error=str(e),
                processing_time=processing_time
            )
    
    def _generate_doc_id(self, source: str, chunk_index: int) -> str:
        """Generate a unique document ID.
        
        Args:
            source: Source identifier
            chunk_index: Chunk index
            
        Returns:
            Unique document ID
        """
        # Create hash from source and chunk index
        content = f"{source}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_processing_stats(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Get statistics from processing results.
        
        Args:
            results: List of processing results
            
        Returns:
            Dictionary containing processing statistics
        """
        total_files = len(results)
        successful_files = sum(1 for r in results if r.success)
        failed_files = total_files - successful_files
        
        total_documents = sum(len(r.documents) for r in results if r.success)
        total_processing_time = sum(r.processing_time for r in results if r.processing_time)
        
        errors = [r.error for r in results if not r.success and r.error]
        
        return {
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': successful_files / total_files if total_files > 0 else 0,
            'total_documents': total_documents,
            'avg_documents_per_file': total_documents / successful_files if successful_files > 0 else 0,
            'total_processing_time': total_processing_time,
            'avg_processing_time': total_processing_time / successful_files if successful_files > 0 else 0,
            'errors': errors
        }

class AdvancedDocumentProcessor(DocumentProcessor):
    """Advanced document processor with additional features."""
    
    def __init__(self, config: DocumentProcessingConfig):
        super().__init__(config)
        self.deduplication_enabled = True
        self.processed_hashes = set()
    
    def enable_deduplication(self, enabled: bool = True):
        """Enable or disable document deduplication.
        
        Args:
            enabled: Whether to enable deduplication
        """
        self.deduplication_enabled = enabled
        if not enabled:
            self.processed_hashes.clear()
    
    def _is_duplicate(self, content: str) -> bool:
        """Check if content is a duplicate.
        
        Args:
            content: Document content to check
            
        Returns:
            True if content is a duplicate
        """
        if not self.deduplication_enabled:
            return False
        
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash in self.processed_hashes:
            return True
        
        self.processed_hashes.add(content_hash)
        return False
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process file with deduplication."""
        result = super().process_file(file_path)
        
        if result.success and self.deduplication_enabled:
            # Filter out duplicates
            unique_documents = []
            for doc in result.documents:
                if not self._is_duplicate(doc.content):
                    unique_documents.append(doc)
                else:
                    logger.debug(f"Skipped duplicate content in {file_path}")
            
            result.documents = unique_documents
        
        return result
    
    def semantic_chunking(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform semantic chunking based on content structure.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata
            
        Returns:
            List of semantically chunked documents
        """
        # This is a simplified semantic chunking implementation
        # In practice, you might want to use more sophisticated methods
        
        import re
        
        # Split by paragraphs and sections
        paragraphs = re.split(r'\n\s*\n', text)
        
        documents = []
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                doc_id = self._generate_doc_id(text[:50], i)
                
                chunk_metadata = {
                    'source': 'semantic_chunking',
                    'chunk_index': i,
                    'total_chunks': len(paragraphs),
                    'chunk_type': 'paragraph'
                }
                
                if metadata:
                    chunk_metadata.update(metadata)
                
                doc = Document(
                    content=paragraph.strip(),
                    metadata=chunk_metadata,
                    doc_id=doc_id,
                    source='semantic_input'
                )
                documents.append(doc)
        
        return documents