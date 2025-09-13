# RAG Agent for Custom Data Analysis

A comprehensive, privacy-focused local RAG (Retrieval-Augmented Generation) agent system that enables you to analyze and query your custom documents using state-of-the-art AI models—all running locally on your machine.

## 🚀 Features

- **🔒 Privacy-First**: All processing happens locally—your data never leaves your machine
- **🤖 Multiple LLM Support**: Integrates with Ollama for local LLM deployment
- **📊 Multi-Format Support**: Process PDF, TXT, MD, CSV, DOCX, and web content
- **🔍 Advanced Retrieval**: Hybrid search with semantic similarity and keyword matching
- **⚡ High Performance**: Optimized vector storage with ChromaDB and intelligent caching
- **🎯 Smart Chunking**: Context-aware document splitting and processing
- **📝 Source Attribution**: Responses include citations to source documents
- **🖥️ CLI Interface**: Easy-to-use command-line interface for all operations
- **🔧 Configurable**: Comprehensive configuration system for all components

## 📋 Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **Minimum 16GB RAM** (32GB recommended for larger documents)
- **50GB+ free disk space** for models and data
- **Ollama** for local LLM deployment

## 🛠️ Quick Start

### 1. Install Ollama

First, install Ollama from [https://ollama.ai](https://ollama.ai):

```bash
# On macOS
brew install ollama

# On Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve
```

### 2. Pull a Language Model

```bash
# For general use (recommended)
ollama pull llama3

# For code-focused tasks
ollama pull codellama

# For resource-constrained environments
ollama pull mistral
```

### 3. Setup RAG Agent

```bash
# Clone the repository
git clone https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis.git
cd Agent-for-Custom-Data-Analysis

# Run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 4. Initialize and Use

```bash
# Initialize the RAG Agent
python main.py init

# Ingest documents from a directory
python main.py ingest /path/to/your/documents

# Start interactive mode
python main.py interactive

# Or query directly
python main.py query "What is the main topic of my documents?"
```

## 📚 Usage Examples

### Command Line Interface

```bash
# Initialize the system
python main.py init

# Ingest a single file
python main.py ingest document.pdf

# Ingest a directory recursively
python main.py ingest -r /path/to/documents

# Query with sources
python main.py query -s "What are the key findings?" --max-sources 3

# Interactive mode
python main.py interactive

# System statistics
python main.py stats

# Backup and restore
python main.py backup my_backup.json
python main.py restore my_backup.json --confirm
```

### Python API

```python
from rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent()
agent.initialize()

# Ingest documents
agent.ingest_directory("./documents")

# Query the agent
response = agent.query("What is machine learning?")
print(response.answer)

# View sources
for source in response.sources:
    print(f"Source: {source['metadata']['file_name']}")
    print(f"Score: {source['similarity_score']}")
```

## 🏗️ Architecture

The RAG Agent follows a modular architecture with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Embedding     │    │   Vector        │
│   Processor     │───▶│   Generator     │───▶│   Store         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG Agent     │◄───│   Document      │◄───│   ChromaDB      │
│   Orchestrator  │    │   Retriever     │    │   (Persistent)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│   LLM Interface │◄───│   Ollama        │
│   (Local)       │    │   Server        │
└─────────────────┘    └─────────────────┘
```

### Core Components

1. **Document Processor**: Handles multiple file formats with intelligent chunking
2. **Embedding Generator**: Supports Ollama, HuggingFace, and SentenceTransformer models
3. **Vector Store**: ChromaDB for efficient similarity search with persistence
4. **Document Retriever**: Advanced retrieval with reranking and hybrid search
5. **LLM Interface**: Ollama integration for local language model inference
6. **RAG Agent**: Main orchestrator coordinating all components

## ⚙️ Configuration

### Default Configuration

The system works out-of-the-box with sensible defaults, but you can customize everything:

```bash
# Generate a configuration template
python main.py generate-config default config.yaml
```

### Key Configuration Options

```yaml
# System settings
system:
  log_level: "INFO"
  cache_size: 1000
  max_query_length: 1000

# Ollama LLM settings
ollama:
  base_url: "http://localhost:11434"
  model_name: "llama3"
  temperature: 0.7
  context_window: 4096

# Embedding settings
embedding:
  model_name: "nomic-embed-text"
  model_type: "ollama"  # or "huggingface", "sentence_transformer"
  chunk_size: 1000
  chunk_overlap: 200

# Document processing
document_processing:
  supported_formats: ["pdf", "txt", "md", "csv", "docx"]
  max_file_size_mb: 100
  parallel_workers: 4

# Retrieval settings
retrieval:
  top_k: 5
  similarity_threshold: 0.7
  enable_reranking: true
```

## 🔧 Advanced Features

### Hybrid Search

Combines semantic similarity with keyword matching:

```python
# Use hybrid search for better results
response = agent.retriever.hybrid_search(
    query="artificial intelligence applications",
    semantic_weight=0.7,
    keyword_weight=0.3
)
```

### Custom Embedding Models

Switch between different embedding models:

```python
# Use HuggingFace embeddings
config.embedding.model_type = "huggingface"
config.embedding.model_name = "all-MiniLM-L6-v2"

# Use Sentence Transformers
config.embedding.model_type = "sentence_transformer"
config.embedding.model_name = "all-mpnet-base-v2"
```

### Performance Optimization

- **Caching**: Query and embedding caching for faster responses
- **Batch Processing**: Efficient batch document processing
- **Parallel Workers**: Configurable parallel processing
- **Memory Management**: Intelligent memory usage optimization

## 🧪 Testing

Run the basic example to test your setup:

```bash
python examples/basic_example.py
```

Expected output:
```
RAG Agent Basic Example
==================================================
1. Initializing RAG Agent...
✓ RAG Agent initialized successfully!

2. Ingesting example documents...
✓ Ingested 5 document chunks

3. Testing queries...
Query 1: What is artificial intelligence?
----------------------------------------
Answer: Artificial Intelligence (AI) is a broad field that encompasses...
Sources: 2
Time: 1.234s
```

## 📊 Performance Benchmarks

Typical performance on a modern machine (16GB RAM, SSD):

- **Document Ingestion**: ~10-50 documents/second
- **Query Response**: 1-3 seconds average
- **Memory Usage**: 2-4GB for typical workloads
- **Storage**: ~100MB per 1000 document chunks

## 🛡️ Security Features

- **Local Processing**: All data stays on your machine
- **No External APIs**: No data sent to external services
- **Content Filtering**: Optional content filtering capabilities
- **Access Control**: File-based access control for sensitive documents

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if not running
   ollama serve
   ```

2. **Model Not Found**
   ```bash
   # Pull the required model
   ollama pull llama3
   ```

3. **Memory Issues**
   ```bash
   # Use a smaller model
   ollama pull mistral:7b
   
   # Or reduce chunk size in config
   chunk_size: 500
   ```

4. **Slow Performance**
   - Reduce `chunk_size` and `top_k` in configuration
   - Use smaller embedding models
   - Enable caching (enabled by default)

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python main.py -v query "your question"
```

## 📖 Documentation Structure

```
docs/
├── installation.md          # Detailed installation guide
├── configuration.md         # Configuration reference
├── api_reference.md         # Python API documentation
├── cli_reference.md         # CLI command reference
├── architecture.md          # System architecture details
└── troubleshooting.md       # Troubleshooting guide
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis.git
cd Agent-for-Custom-Data-Analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local LLM deployment
- [LangChain](https://langchain.com) for document processing framework
- [ChromaDB](https://www.trychroma.com) for vector database
- [Sentence Transformers](https://www.sbert.net) for embedding models

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis/discussions)
- **Documentation**: [Wiki](https://github.com/Maimoon-github/Agent-for-Custom-Data-Analysis/wiki)

---

**Built with ❤️ for privacy-conscious AI applications**