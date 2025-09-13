# RAG Agent for Custom Data Analysis

A privacy-focused, local RAG (Retrieval-Augmented Generation) system that enables intelligent document analysis and question-answering without sending data to external APIs. Built with LangChain, ChromaDB, and Ollama for complete local deployment.

## 🌟 Features

### 🔐 Privacy-First Design
- **100% Local Processing**: All data stays on your machine
- **No External API Calls**: Uses local Ollama models
- **Secure Document Storage**: ChromaDB vector database with local persistence
- **Data Encryption**: Optional encryption for sensitive documents

### 📚 Comprehensive Document Support
- **Multiple Formats**: PDF, TXT, DOCX, CSV files
- **Smart Chunking**: Adaptive chunking strategies based on document type
- **Batch Processing**: Efficient handling of large document collections
- **Metadata Extraction**: Automatic extraction of document metadata

### 🧠 Advanced RAG Capabilities
- **Semantic Search**: Vector similarity search with embedding models
- **Hybrid Retrieval**: Combines semantic and keyword-based search
- **Context Optimization**: Dynamic context preparation for better responses
- **Source Citations**: Automatic citation of sources in responses

### 🎯 Intelligent Query Processing
- **Auto Query Classification**: Automatic detection of query types
- **Multiple Response Modes**: Factual, analytical, comparative, and summary modes
- **Confidence Scoring**: AI confidence assessment for responses
- **Response Reranking**: Quality-based document reranking

### 🚀 Multiple Interfaces
- **Command Line Interface**: Rich CLI with interactive mode
- **Web Application**: Modern Streamlit-based web interface
- **API Integration**: Easy integration with other applications

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │  RAG Agent      │    │   Ollama LLM    │
│   (PDF, TXT,    │──→ │  (Orchestrator) │──→ │  (Local Model)  │
│    DOCX, CSV)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Document Processor│    │ Vector Database │    │Response Generator│
│  - Chunking     │    │   (ChromaDB)    │    │ - Prompt Eng.  │
│  - Validation   │    │   - Storage     │    │ - Source Cite  │
│  - Metadata     │    │   - Search      │    │ - Post-process │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- 16GB+ RAM (32GB recommended)
- 50GB+ free disk space
- Optional: CUDA-capable GPU for faster inference

### Step 1: Install Ollama
Download and install Ollama from [https://ollama.ai](https://ollama.ai)

**Windows:**
```bash
# Download installer from https://ollama.ai/download
# Run the installer
```

**macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Pull a Language Model
```bash
# For general use (recommended)
ollama pull llama3

# For code-focused tasks
ollama pull codellama

# For resource-constrained systems
ollama pull mistral

# For multilingual support
ollama pull llama3:70b
```

### Step 3: Clone and Setup RAG Agent
```bash
git clone https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis.git
cd Agent-for-Custom-Data-Analysis

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate rag-agent
```

### Step 4: Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env
```

## 🚀 Quick Start

### Command Line Interface
```bash
# Start interactive mode
python interfaces/cli.py

# Add documents
python interfaces/cli.py --add-docs /path/to/your/documents

# Ask a single question
python interfaces/cli.py --query "What are the main findings in the research papers?"

# Check system health
python interfaces/cli.py --health
```

### Web Interface
```bash
# Start the web application
streamlit run interfaces/web_app.py

# Open browser to http://localhost:8501
```

### Programmatic Usage
```python
from src.core.rag_agent import rag_agent

# Add documents
result = rag_agent.add_documents("/path/to/documents")
print(f"Added {result['documents_added']} documents")

# Query the system
response = rag_agent.query("What are the key trends in the data?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence_score:.1%}")
print(f"Sources: {len(response.sources)}")
```

## 📖 Usage Examples

### Document Analysis
```python
# Add various document types
rag_agent.add_documents("./data/research_papers", recursive=True)

# Analyze findings
response = rag_agent.query(
    "What are the main research findings across all papers?",
    query_type="analytical"
)

# Compare approaches
response = rag_agent.query(
    "Compare the methodologies used in different studies",
    query_type="comparative"
)
```

### Business Intelligence
```python
# Load business reports
rag_agent.add_documents("./reports/quarterly_reports.pdf")

# Financial analysis
response = rag_agent.query(
    "What are the key financial performance indicators?",
    query_type="factual"
)

# Trend analysis
response = rag_agent.query(
    "Analyze the revenue trends over the past quarters",
    query_type="analytical"
)
```

### Data Exploration
```python
# Load datasets
rag_agent.add_documents("./data/sales_data.csv")

# Summarize data
response = rag_agent.query(
    "Provide a summary of the sales data",
    query_type="summary"
)

# Pattern recognition
response = rag_agent.query(
    "What patterns can you identify in customer behavior?",
    query_type="pattern"
)
```

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
TEMPERATURE=0.7
MAX_TOKENS=2048

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNK_SIZE=1500

# Retrieval Settings
RETRIEVAL_K=5
SIMILARITY_THRESHOLD=0.7
RERANK_TOP_K=3

# Vector Database
CHROMA_DB_PATH=./data/vectordb
COLLECTION_NAME=knowledge_base

# Performance
BATCH_SIZE=100
MAX_WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/rag_agent.log
```

### Model Selection Guide

| Use Case | Recommended Model | RAM Required | Quality |
|----------|------------------|--------------|---------|
| General QA | llama3 | 8GB | High |
| Code Analysis | codellama | 8GB | High |
| Resource Constrained | mistral | 4GB | Good |
| Research Papers | llama3:70b | 32GB | Excellent |
| Multilingual | llama3 | 8GB | High |

## 📊 Performance Optimization

### Hardware Recommendations
- **CPU**: 8+ cores for optimal performance
- **RAM**: 16GB minimum, 32GB+ for large datasets
- **Storage**: SSD for vector database storage
- **GPU**: Optional CUDA GPU for 2-3x speedup

### Tuning Parameters
```python
# For faster responses (lower quality)
config = {
    "chunk_size": 500,
    "retrieval_k": 3,
    "temperature": 0.3
}

# For better quality (slower responses)
config = {
    "chunk_size": 1500,
    "retrieval_k": 10,
    "temperature": 0.7,
    "rerank_top_k": 5
}
```

## 🔧 Advanced Features

### Custom Embeddings
```python
# Use different embedding models
from sentence_transformers import SentenceTransformer

# For multilingual documents
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# For domain-specific documents
model = SentenceTransformer('allenai-specter')
```

### Custom Prompt Templates
```python
from src.core.response_generator import response_generator

# Add custom prompt template
custom_template = """
Context: {context}
Question: {question}

Please provide a detailed technical analysis with:
1. Key findings
2. Methodology assessment
3. Limitations
4. Recommendations

Analysis:
"""

response_generator.prompt_template.templates['technical'] = custom_template
```

### Batch Processing
```python
# Process multiple queries
queries = [
    "What are the main findings?",
    "What methodology was used?",
    "What are the limitations?"
]

results = []
for query in queries:
    response = rag_agent.query(query)
    results.append(response)
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_document_processor.py

# Run with coverage
pytest --cov=src tests/
```

### Integration Tests
```bash
# Test full pipeline
python tests/integration/test_full_pipeline.py

# Test with sample documents
python tests/integration/test_sample_documents.py
```

### Performance Benchmarks
```bash
# Run performance tests
python tests/benchmarks/benchmark_retrieval.py
python tests/benchmarks/benchmark_generation.py
```

## 🔍 Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Check if Ollama is running
ollama --version

# Start Ollama service
ollama serve

# Check available models
ollama list
```

**Out of Memory Errors**
```bash
# Reduce chunk size
export CHUNK_SIZE=500

# Use smaller model
ollama pull mistral

# Reduce context window
export CONTEXT_WINDOW=2048
```

**Slow Performance**
- Enable GPU acceleration
- Reduce document chunk size
- Use faster embedding model
- Implement document caching

**Poor Retrieval Quality**
- Increase chunk overlap
- Adjust similarity threshold
- Use hybrid retrieval mode
- Improve document preprocessing

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python interfaces/cli.py --verbose
```

## 📝 API Reference

### RAG Agent
```python
class RAGAgent:
    def add_documents(source_path: str, recursive: bool = True) -> Dict
    def query(question: str, query_type: str = 'general', **kwargs) -> RAGResponse
    def search_knowledge_base(query: str, k: int = 10) -> List[Dict]
    def get_knowledge_base_stats() -> Dict
    def health_check() -> Dict
    def reset_knowledge_base() -> bool
```

### Document Processor
```python
class DocumentProcessor:
    def load_document(file_path: str) -> List[Document]
    def load_directory(directory_path: str, recursive: bool = True) -> List[Document]
    def validate_documents(documents: List[Document]) -> List[Document]
```

### Vector Database
```python
class VectorDatabase:
    def add_documents(documents: List[Document]) -> Dict
    def search(query: str, k: int, filter_dict: Dict = None) -> List[Tuple[Document, float]]
    def delete_documents(filter_dict: Dict = None, ids: List[str] = None) -> int
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis.git
cd Agent-for-Custom-Data-Analysis

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector database
- [Ollama](https://ollama.ai/) for local LLM deployment
- [Streamlit](https://streamlit.io/) for web interface
- [Rich](https://rich.readthedocs.io/) for CLI interface

## 📞 Support

- 📫 **Issues**: [GitHub Issues](https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis/wiki)

## 🗺️ Roadmap

- [ ] **Multi-modal Support**: Images, audio, video analysis
- [ ] **Advanced Analytics**: Statistical analysis integration
- [ ] **Collaborative Features**: Multi-user knowledge bases
- [ ] **API Server**: RESTful API for integration
- [ ] **Mobile App**: Mobile interface for queries
- [ ] **Plugin System**: Extensible plugin architecture
- [ ] **Cloud Deployment**: Optional cloud deployment guides
- [ ] **Model Fine-tuning**: Domain-specific model adaptation

---

**Built with ❤️ for privacy-conscious data analysis**