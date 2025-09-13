"""
Command-line interface for the RAG Agent.
Provides interactive and batch operations.
"""

import click
import json
import time
import yaml
from pathlib import Path
from typing import Optional

from .rag_agent import RAGAgent
from .config import RAGConfig

# Global agent instance
agent = None

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(config: Optional[str], verbose: bool):
    """RAG Agent - Privacy-focused local document analysis system."""
    global agent
    
    # Load configuration
    rag_config = RAGConfig(config_file=config) if config else RAGConfig()
    
    if verbose:
        rag_config.system.log_level = "DEBUG"
    
    # Initialize agent
    agent = RAGAgent(rag_config)
    
    click.echo("RAG Agent CLI initialized")

@cli.command()
def init():
    """Initialize the RAG Agent system."""
    click.echo("Initializing RAG Agent...")
    
    with click.progressbar(length=100, label='Initializing components') as bar:
        if agent.initialize():
            bar.update(100)
            click.echo("✓ RAG Agent initialized successfully!")
        else:
            click.echo("✗ Failed to initialize RAG Agent. Check logs for details.")
            return 1

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
def ingest(path: str, recursive: bool):
    """Ingest documents from a file or directory."""
    if not agent.is_initialized:
        click.echo("Error: RAG Agent not initialized. Run 'init' command first.")
        return 1
    
    path_obj = Path(path)
    
    click.echo(f"Ingesting: {path}")
    
    if path_obj.is_file():
        # Ingest single file
        with click.progressbar(length=1, label='Processing file') as bar:
            success = agent.ingest_document(path)
            bar.update(1)
            
        if success:
            click.echo("✓ File ingested successfully!")
        else:
            click.echo("✗ Failed to ingest file. Check logs for details.")
            return 1
            
    elif path_obj.is_dir():
        # Ingest directory
        click.echo("Processing directory...")
        stats = agent.ingest_directory(path, recursive=recursive)
        
        if stats['success']:
            click.echo(f"✓ Directory ingested successfully!")
            click.echo(f"  - Files processed: {stats['total_files']}")
            click.echo(f"  - Successful: {stats['successful_files']}")
            click.echo(f"  - Failed: {stats['failed_files']}")
            click.echo(f"  - Total documents: {stats['total_documents']}")
            click.echo(f"  - Processing time: {stats['total_processing_time']:.2f}s")
        else:
            click.echo(f"✗ Failed to ingest directory: {stats.get('error', 'Unknown error')}")
            return 1
    else:
        click.echo("Error: Path is neither a file nor a directory.")
        return 1

@cli.command()
@click.argument('question')
@click.option('--sources', '-s', is_flag=True, help='Include source documents in response')
@click.option('--max-sources', '-m', type=int, default=5, help='Maximum number of sources to retrieve')
@click.option('--output', '-o', type=click.Path(), help='Save response to file')
def query(question: str, sources: bool, max_sources: int, output: Optional[str]):
    """Query the RAG agent with a question."""
    if not agent.is_initialized:
        click.echo("Error: RAG Agent not initialized. Run 'init' command first.")
        return 1
    
    click.echo(f"Query: {question}")
    click.echo("Searching...")
    
    # Query the agent
    response = agent.query(
        question=question,
        include_sources=sources,
        max_sources=max_sources
    )
    
    # Display response
    click.echo("\n" + "="*50)
    click.echo("ANSWER:")
    click.echo("="*50)
    click.echo(response.answer)
    
    if sources and response.sources:
        click.echo("\n" + "="*50)
        click.echo("SOURCES:")
        click.echo("="*50)
        for i, source in enumerate(response.sources):
            click.echo(f"\nSource {i+1} (Score: {source['similarity_score']}):")
            click.echo(f"File: {source['metadata']['file_name']}")
            click.echo(f"Type: {source['metadata']['file_type']}")
            click.echo(f"Content: {source['content']}")
    
    # Show statistics
    click.echo(f"\nResponse time: {response.response_time:.3f}s")
    click.echo(f"Sources used: {response.num_sources_used}")
    
    # Save to file if requested
    if output:
        response_data = {
            'query': response.query,
            'answer': response.answer,
            'sources': response.sources,
            'response_time': response.response_time,
            'num_sources_used': response.num_sources_used,
            'retrieval_stats': response.retrieval_stats
        }
        
        with open(output, 'w') as f:
            json.dump(response_data, f, indent=2)
        
        click.echo(f"Response saved to: {output}")

@cli.command()
def interactive():
    """Start interactive query mode."""
    if not agent.is_initialized:
        click.echo("Error: RAG Agent not initialized. Run 'init' command first.")
        return 1
    
    click.echo("RAG Agent Interactive Mode")
    click.echo("Type 'quit' or 'exit' to leave, 'help' for commands")
    click.echo("="*50)
    
    while True:
        try:
            question = click.prompt("\nQuestion", type=str)
            
            if question.lower() in ['quit', 'exit']:
                break
            elif question.lower() == 'help':
                click.echo("\nCommands:")
                click.echo("  help - Show this help")
                click.echo("  stats - Show system statistics")
                click.echo("  history - Show conversation history")
                click.echo("  clear - Clear conversation history")
                click.echo("  quit/exit - Exit interactive mode")
                continue
            elif question.lower() == 'stats':
                stats = agent.get_system_stats()
                click.echo(json.dumps(stats, indent=2))
                continue
            elif question.lower() == 'history':
                history = agent.get_conversation_history()
                for i, conv in enumerate(history):
                    click.echo(f"\n{i+1}. Q: {conv['query']}")
                    click.echo(f"   A: {conv['answer'][:100]}...")
                continue
            elif question.lower() == 'clear':
                agent.clear_conversation_history()
                click.echo("Conversation history cleared.")
                continue
            
            # Process question
            response = agent.query(question, include_sources=True)
            
            click.echo(f"\nAnswer: {response.answer}")
            
            if response.sources:
                click.echo(f"\nSources ({len(response.sources)}):")
                for i, source in enumerate(response.sources):
                    click.echo(f"  {i+1}. {source['metadata']['file_name']} (Score: {source['similarity_score']})")
            
            click.echo(f"\nTime: {response.response_time:.3f}s")
            
        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"Error: {e}")

@cli.command()
def stats():
    """Show system statistics."""
    if not agent.is_initialized:
        click.echo("Error: RAG Agent not initialized. Run 'init' command first.")
        return 1
    
    stats = agent.get_system_stats()
    
    click.echo("RAG Agent Statistics")
    click.echo("="*50)
    
    # Vector store stats
    if 'vector_store' in stats:
        vs_stats = stats['vector_store']
        click.echo(f"Documents in database: {vs_stats.get('total_documents', 0)}")
        click.echo(f"Collection name: {vs_stats.get('collection_name', 'N/A')}")
    
    # Conversation stats
    click.echo(f"Conversations: {stats.get('conversation_count', 0)}")
    
    # Retrieval stats
    if 'retrieval' in stats:
        ret_stats = stats['retrieval']
        click.echo(f"Total queries: {ret_stats.get('total_queries', 0)}")
        click.echo(f"Avg retrieval time: {ret_stats.get('avg_retrieval_time', 0):.3f}s")
        click.echo(f"Avg results per query: {ret_stats.get('avg_results_per_query', 0):.1f}")
    
    # Embedding cache stats
    if 'embedding_cache' in stats:
        cache_stats = stats['embedding_cache']
        click.echo(f"Cache size: {cache_stats.get('cache_size', 0)}")
        click.echo(f"Cache utilization: {cache_stats.get('cache_utilization', 0):.1%}")

@cli.command()
@click.argument('backup_path', type=click.Path())
def backup(backup_path: str):
    """Backup the vector database."""
    if not agent.is_initialized:
        click.echo("Error: RAG Agent not initialized. Run 'init' command first.")
        return 1
    
    click.echo(f"Creating backup: {backup_path}")
    
    if agent.backup_data(backup_path):
        click.echo("✓ Backup created successfully!")
    else:
        click.echo("✗ Failed to create backup. Check logs for details.")
        return 1

@cli.command()
@click.argument('backup_path', type=click.Path(exists=True))
@click.option('--confirm', is_flag=True, help='Confirm restoration (will overwrite existing data)')
def restore(backup_path: str, confirm: bool):
    """Restore the vector database from backup."""
    if not agent.is_initialized:
        click.echo("Error: RAG Agent not initialized. Run 'init' command first.")
        return 1
    
    if not confirm:
        click.echo("Warning: This will overwrite existing data.")
        if not click.confirm("Do you want to continue?"):
            click.echo("Restoration cancelled.")
            return 0
    
    click.echo(f"Restoring from backup: {backup_path}")
    
    if agent.restore_data(backup_path):
        click.echo("✓ Data restored successfully!")
    else:
        click.echo("✗ Failed to restore data. Check logs for details.")
        return 1

@cli.command()
@click.option('--confirm', is_flag=True, help='Confirm reset (will delete all data)')
def reset(confirm: bool):
    """Reset the vector database (delete all data)."""
    if not agent.is_initialized:
        click.echo("Error: RAG Agent not initialized. Run 'init' command first.")
        return 1
    
    if not confirm:
        click.echo("Warning: This will delete all ingested documents.")
        if not click.confirm("Do you want to continue?"):
            click.echo("Reset cancelled.")
            return 0
    
    click.echo("Resetting vector database...")
    
    if agent.reset_vector_store():
        click.echo("✓ Vector database reset successfully!")
    else:
        click.echo("✗ Failed to reset database. Check logs for details.")
        return 1

@cli.command()
@click.option('--template', type=click.Choice(['default', 'minimal', 'advanced']), default='default')
@click.argument('output_path', type=click.Path())
def generate_config(template: str, output_path: str):
    """Generate a configuration file template."""
    
    if template == 'minimal':
        config_content = """# Minimal RAG Agent Configuration
system:
  log_level: "INFO"
  
ollama:
  model_name: "llama3"
  
embedding:
  model_name: "nomic-embed-text"
  model_type: "ollama"
"""
    elif template == 'advanced':
        # For advanced, just show the comprehensive structure
        config_content = """# Advanced RAG Agent Configuration
# This shows all available configuration options

system:
  log_level: "INFO"
  data_dir: "./data"
  logs_dir: "./logs"
  temp_dir: "./temp"
  cache_size: 1000
  cache_ttl_seconds: 3600
  max_query_length: 1000
  enable_content_filtering: true

ollama:
  base_url: "http://localhost:11434"
  model_name: "llama3"
  temperature: 0.7
  context_window: 4096
  num_gpu: 1
  timeout: 300

chromadb:
  persist_directory: "./chroma_db"
  collection_name: "domain_knowledge"
  distance_metric: "cosine"
  max_batch_size: 100

embedding:
  model_name: "nomic-embed-text"
  model_type: "ollama"
  chunk_size: 1000
  chunk_overlap: 200
  max_tokens: 8192

document_processing:
  supported_formats: ["pdf", "txt", "md", "csv", "docx"]
  max_file_size_mb: 100
  batch_size: 50
  parallel_workers: 4

retrieval:
  top_k: 5
  similarity_threshold: 0.7
  rerank_top_k: 10
  enable_reranking: true
"""
    else:  # default
        config_content = """# Default RAG Agent Configuration
system:
  log_level: "INFO"
  data_dir: "./data"
  logs_dir: "./logs"
  cache_size: 1000

ollama:
  base_url: "http://localhost:11434"
  model_name: "llama3"
  temperature: 0.7

embedding:
  model_name: "nomic-embed-text"
  model_type: "ollama"
  chunk_size: 1000
  chunk_overlap: 200

retrieval:
  top_k: 5
  similarity_threshold: 0.7
  enable_reranking: true
"""
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    click.echo(f"Configuration template saved to: {output_path}")

@cli.command()
def version():
    """Show version information."""
    from . import __version__, __description__
    click.echo(f"RAG Agent v{__version__}")
    click.echo(__description__)

if __name__ == '__main__':
    cli()