"""
Document ingestion pipeline for the RAG system
Supports PDF, TXT, DOCX, and CSV files with smart chunking strategies
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from langchain.schema import Document

from config.settings import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    created_at: datetime
    modified_at: datetime
    document_type: str
    chunk_count: int
    processing_method: str

class DocumentProcessor:
    """Handles document loading, preprocessing, and chunking"""
    
    def __init__(self):
        self.config = config.document
        self.supported_loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.docx': UnstructuredWordDocumentLoader
        }
        
        # Initialize text splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash for file content"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract file metadata"""
        path_obj = Path(file_path)
        stat_info = path_obj.stat()
        
        return {
            'file_path': str(path_obj.absolute()),
            'file_name': path_obj.name,
            'file_size': stat_info.st_size,
            'file_hash': self._get_file_hash(file_path),
            'created_at': datetime.fromtimestamp(stat_info.st_ctime),
            'modified_at': datetime.fromtimestamp(stat_info.st_mtime),
            'document_type': path_obj.suffix.lower()
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess document text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove page headers/footers patterns (basic implementation)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely headers/footers (very short lines, page numbers, etc.)
            if len(line) < 3 or line.isdigit():
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _choose_splitting_strategy(self, document: Document, file_extension: str) -> str:
        """Determine optimal splitting strategy based on document characteristics"""
        content_length = len(document.page_content)
        
        # For very large documents, use token-based splitting
        if content_length > 10000:
            return "token_based"
        
        # For structured documents (CSV), use semantic splitting
        if file_extension == '.csv':
            return "semantic"
        
        # Default to recursive character splitting
        return "recursive"
    
    def _split_document(self, document: Document, strategy: str = "recursive") -> List[Document]:
        """Split document into chunks using specified strategy"""
        if strategy == "token_based":
            return self.token_splitter.split_documents([document])
        else:  # Default to recursive
            return self.recursive_splitter.split_documents([document])
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document and return processed chunks"""
        try:
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = path_obj.suffix.lower()
            
            if file_extension not in self.supported_loaders:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Get file metadata
            file_metadata = self._get_file_metadata(file_path)
            
            # Load document using appropriate loader
            loader_class = self.supported_loaders[file_extension]
            
            if file_extension == '.csv':
                # For CSV files, specify encoding and handle potential issues
                loader = loader_class(file_path, encoding='utf-8')
            else:
                loader = loader_class(file_path)
            
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No content loaded from {file_path}")
                return []
            
            # Preprocess content
            processed_docs = []
            for doc in documents:
                # Clean text content
                cleaned_content = self._preprocess_text(doc.page_content)
                
                # Update document with cleaned content and metadata
                doc.page_content = cleaned_content
                doc.metadata.update(file_metadata)
                
                processed_docs.append(doc)
            
            # Choose splitting strategy
            strategy = self._choose_splitting_strategy(processed_docs[0], file_extension)
            
            # Split documents into chunks
            all_chunks = []
            for doc in processed_docs:
                chunks = self._split_document(doc, strategy)
                
                # Add chunk-specific metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_id': f"{file_metadata['file_hash']}_{i}",
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'processing_method': strategy,
                        'chunk_size': len(chunk.page_content)
                    })
                
                all_chunks.extend(chunks)
            
            # Update metadata with final chunk count
            for chunk in all_chunks:
                chunk.metadata['chunk_count'] = len(all_chunks)
            
            logger.info(f"Successfully processed {file_path}: {len(all_chunks)} chunks created")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """Load all supported documents from a directory"""
        try:
            directory = Path(directory_path)
            
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"Invalid directory: {directory_path}")
            
            all_documents = []
            pattern = "**/*" if recursive else "*"
            
            for file_path in directory.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_loaders:
                    try:
                        documents = self.load_document(str(file_path))
                        all_documents.extend(documents)
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {str(e)}")
                        continue
            
            logger.info(f"Loaded {len(all_documents)} document chunks from {directory_path}")
            return all_documents
            
        except Exception as e:
            logger.error(f"Error loading directory {directory_path}: {str(e)}")
            raise
    
    def validate_documents(self, documents: List[Document]) -> List[Document]:
        """Validate and filter processed documents"""
        valid_documents = []
        
        for doc in documents:
            # Check minimum content length
            if len(doc.page_content.strip()) < 10:
                logger.warning(f"Skipping document chunk with insufficient content")
                continue
            
            # Check maximum chunk size
            if len(doc.page_content) > self.config.max_chunk_size:
                logger.warning(f"Chunk exceeds maximum size, truncating")
                doc.page_content = doc.page_content[:self.config.max_chunk_size]
            
            valid_documents.append(doc)
        
        logger.info(f"Validated {len(valid_documents)} documents")
        return valid_documents
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate statistics about processed documents"""
        if not documents:
            return {}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        file_types = {}
        
        for doc in documents:
            doc_type = doc.metadata.get('document_type', 'unknown')
            file_types[doc_type] = file_types.get(doc_type, 0) + 1
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'average_chunk_size': total_chars / len(documents),
            'file_types': file_types,
            'unique_files': len(set(doc.metadata.get('file_path', '') for doc in documents))
        }

# Create global processor instance
document_processor = DocumentProcessor()
