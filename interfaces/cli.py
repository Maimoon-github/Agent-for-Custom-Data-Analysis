"""
Command Line Interface for the RAG Agent
Provides interactive and batch modes for document analysis
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from src.core.rag_agent import rag_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)
console = Console()

class RAGCLI:
    """Command Line Interface for the RAG Agent"""
    
    def __init__(self):
        self.agent = rag_agent
        self.session_history = []
    
    def display_banner(self):
        """Display the application banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                   RAG Agent for Custom Data Analysis        ║
║                  Privacy-Focused Local AI Assistant         ║
╚══════════════════════════════════════════════════════════════╝
        """
        console.print(banner, style="bold blue")
    
    def interactive_mode(self):
        """Run the CLI in interactive mode"""
        self.display_banner()
        
        # Perform health check
        console.print("\n🔍 Performing system health check...", style="yellow")
        health = self.agent.health_check()
        
        if health["status"] == "healthy":
            console.print("✅ System is healthy and ready!", style="green")
        elif health["status"] == "degraded":
            console.print("⚠️  System is partially functional", style="yellow")
            self._display_health_details(health)
        else:
            console.print("❌ System is unhealthy", style="red")
            self._display_health_details(health)
            if not Confirm.ask("Continue anyway?"):
                return
        
        # Display knowledge base stats
        stats = self.agent.get_knowledge_base_stats()
        kb_docs = stats.get("knowledge_base", {}).get("total_documents", 0)
        console.print(f"\n📚 Knowledge Base: {kb_docs} documents loaded", style="cyan")
        
        if kb_docs == 0:
            if Confirm.ask("\nNo documents in knowledge base. Would you like to add some?"):
                self.add_documents_interactive()
        
        console.print("\n" + "="*60)
        console.print("Interactive RAG Agent - Type 'help' for commands or 'quit' to exit")
        console.print("="*60)
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]RAG>[/bold cyan]", default="").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 Goodbye!", style="green")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower().startswith('add '):
                    path = user_input[4:].strip()
                    self.add_documents_batch(path)
                elif user_input.lower() == 'stats':
                    self._show_stats()
                elif user_input.lower() == 'search':
                    self._interactive_search()
                elif user_input.lower() == 'health':
                    self._show_health()
                elif user_input.lower() == 'history':
                    self._show_history()
                elif user_input.lower() == 'clear':
                    console.clear()
                elif user_input.lower().startswith('save '):
                    filename = user_input[5:].strip()
                    self._save_session(filename)
                else:
                    # Treat as a query
                    self._process_query(user_input)
                    
            except KeyboardInterrupt:
                console.print("\n\n👋 Goodbye!", style="green")
                break
            except Exception as e:
                console.print(f"\n❌ Error: {str(e)}", style="red")
                logger.error(f"CLI error: {str(e)}")
    
    def _process_query(self, question: str):
        """Process a user query and display results"""
        console.print(f"\n🤔 Processing: [italic]{question}[/italic]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching knowledge base...", total=None)
            
            try:
                # Get response from RAG agent
                response = self.agent.query(question, query_type='auto')
                
                progress.update(task, description="Generating response...")
                
                # Display the response
                self._display_response(response)
                
                # Add to session history
                self.session_history.append({
                    'timestamp': response.metadata['timestamp'],
                    'question': question,
                    'answer': response.answer,
                    'confidence': response.confidence_score,
                    'sources_count': len(response.sources),
                    'processing_time': response.processing_time
                })
                
                progress.remove_task(task)
                
            except Exception as e:
                progress.remove_task(task)
                console.print(f"\n❌ Error processing query: {str(e)}", style="red")
    
    def _display_response(self, response):
        """Display the RAG response in a formatted way"""
        # Main answer
        answer_panel = Panel(
            Markdown(response.answer),
            title=f"Answer (Confidence: {response.confidence_score:.1%})",
            border_style="green" if response.confidence_score > 0.7 else "yellow"
        )
        console.print(answer_panel)
        
        # Sources
        if response.sources:
            sources_table = Table(title="Sources Used", show_header=True)
            sources_table.add_column("Rank", style="cyan", width=6)
            sources_table.add_column("Document", style="blue")
            sources_table.add_column("Type", style="magenta", width=8)
            sources_table.add_column("Relevance", style="green", width=10)
            
            for source in response.sources:
                sources_table.add_row(
                    str(source['rank']),
                    source['filename'],
                    source.get('document_type', 'Unknown'),
                    f"{source['relevance_score']:.3f}"
                )
            
            console.print(sources_table)
        
        # Performance stats
        stats_text = (
            f"⏱️  Processing Time: {response.processing_time:.2f}s | "
            f"📄 Documents Found: {response.retrieval_stats['documents_found']} | "
            f"🎯 Documents Used: {response.retrieval_stats['documents_used']}"
        )
        console.print(f"\n{stats_text}", style="dim")
    
    def _show_help(self):
        """Display help information"""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[bold]Query Commands:[/bold]
  • Just type your question to search the knowledge base
  • The system will automatically determine the best approach

[bold]Document Management:[/bold]
  • [yellow]add <path>[/yellow]     - Add documents from file or directory
  • [yellow]stats[/yellow]          - Show knowledge base statistics
  • [yellow]search[/yellow]         - Interactive document search

[bold]System Commands:[/bold]
  • [yellow]health[/yellow]         - Check system health
  • [yellow]history[/yellow]        - Show query history
  • [yellow]clear[/yellow]          - Clear screen
  • [yellow]save <file>[/yellow]    - Save session to file
  • [yellow]help[/yellow]           - Show this help
  • [yellow]quit[/yellow] or [yellow]exit[/yellow] - Exit the application

[bold]Example Queries:[/bold]
  • "What are the main trends in the sales data?"
  • "Explain the key findings from the research papers"
  • "Compare performance metrics across different regions"
  • "Summarize the quarterly financial results"
        """
        console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def _show_stats(self):
        """Display knowledge base statistics"""
        stats = self.agent.get_knowledge_base_stats()
        
        # Knowledge base stats
        kb_stats = stats.get("knowledge_base", {})
        
        stats_table = Table(title="Knowledge Base Statistics", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Documents", str(kb_stats.get("total_documents", 0)))
        stats_table.add_row("Unique Sources", str(kb_stats.get("unique_sources", 0)))
        
        file_types = kb_stats.get("file_types", {})
        for file_type, count in file_types.items():
            stats_table.add_row(f"  {file_type} files", str(count))
        
        console.print(stats_table)
        
        # System configuration
        config_stats = stats.get("configuration", {})
        config_text = (
            f"Chunk Size: {config_stats.get('chunk_size', 'Unknown')} | "
            f"Retrieval K: {config_stats.get('retrieval_k', 'Unknown')} | "
            f"Embedding Model: {config_stats.get('embedding_model', 'Unknown')}"
        )
        console.print(f"\n[dim]Configuration: {config_text}[/dim]")
    
    def _show_health(self):
        """Display system health information"""
        health = self.agent.health_check()
        self._display_health_details(health)
    
    def _display_health_details(self, health):
        """Display detailed health information"""
        health_table = Table(title="System Health", show_header=True)
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="bold")
        health_table.add_column("Details", style="dim")
        
        for component, details in health.get("components", {}).items():
            status = details["status"]
            status_style = "green" if status == "healthy" else "red"
            
            detail_text = ""
            if component == "ollama":
                detail_text = f"Model Available: {details.get('model_available', False)}"
            elif component == "vector_database":
                detail_text = f"Documents: {details.get('document_count', 0)}"
            
            health_table.add_row(
                component.replace("_", " ").title(),
                f"[{status_style}]{status.upper()}[/{status_style}]",
                detail_text
            )
        
        console.print(health_table)
    
    def _show_history(self):
        """Display query history"""
        if not self.session_history:
            console.print("No queries in this session yet.", style="yellow")
            return
        
        history_table = Table(title="Query History", show_header=True)
        history_table.add_column("#", style="cyan", width=3)
        history_table.add_column("Question", style="blue")
        history_table.add_column("Confidence", style="green", width=10)
        history_table.add_column("Sources", style="magenta", width=8)
        history_table.add_column("Time", style="yellow", width=8)
        
        for i, entry in enumerate(self.session_history[-10:], 1):  # Show last 10
            history_table.add_row(
                str(i),
                entry['question'][:50] + "..." if len(entry['question']) > 50 else entry['question'],
                f"{entry['confidence']:.1%}",
                str(entry['sources_count']),
                f"{entry['processing_time']:.1f}s"
            )
        
        console.print(history_table)
    
    def _interactive_search(self):
        """Interactive document search"""
        query = Prompt.ask("Enter search query")
        k = int(Prompt.ask("Number of results", default="5"))
        
        results = self.agent.search_knowledge_base(query, k=k)
        
        if not results:
            console.print("No results found.", style="yellow")
            return
        
        for i, result in enumerate(results, 1):
            result_panel = Panel(
                f"[bold]Score:[/bold] {result['score']:.3f}\n"
                f"[bold]Source:[/bold] {result['source']}\n"
                f"[bold]Type:[/bold] {result['document_type']}\n\n"
                f"{result['content'][:300]}{'...' if len(result['content']) > 300 else ''}",
                title=f"Result {i}",
                border_style="blue"
            )
            console.print(result_panel)
    
    def _save_session(self, filename):
        """Save session history to file"""
        try:
            session_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'history': self.session_history,
                'stats': self.agent.get_knowledge_base_stats()
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            console.print(f"✅ Session saved to {filename}", style="green")
            
        except Exception as e:
            console.print(f"❌ Error saving session: {str(e)}", style="red")
    
    def add_documents_interactive(self):
        """Interactive document addition"""
        path = Prompt.ask("Enter path to documents (file or directory)")
        
        if not Path(path).exists():
            console.print("❌ Path does not exist", style="red")
            return
        
        recursive = True
        if Path(path).is_dir():
            recursive = Confirm.ask("Include subdirectories?", default=True)
        
        self.add_documents_batch(path, recursive)
    
    def add_documents_batch(self, path: str, recursive: bool = True):
        """Add documents in batch mode"""
        console.print(f"\n📂 Adding documents from: {path}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing documents...", total=None)
            
            try:
                result = self.agent.add_documents(path, recursive=recursive)
                progress.remove_task(task)
                
                if result["status"] == "success":
                    console.print(
                        f"✅ Successfully added {result['documents_added']} documents "
                        f"({result['documents_processed']} processed, "
                        f"{result['documents_failed']} failed) "
                        f"in {result['processing_time']:.2f}s",
                        style="green"
                    )
                    
                    # Show document stats
                    doc_stats = result.get('document_stats', {})
                    if doc_stats:
                        stats_text = (
                            f"Total characters: {doc_stats.get('total_characters', 0):,} | "
                            f"Average chunk size: {doc_stats.get('average_chunk_size', 0):.0f} | "
                            f"File types: {list(doc_stats.get('file_types', {}).keys())}"
                        )
                        console.print(f"[dim]{stats_text}[/dim]")
                else:
                    console.print(f"⚠️  {result.get('message', 'Unknown error')}", style="yellow")
                    
            except Exception as e:
                progress.remove_task(task)
                console.print(f"❌ Error adding documents: {str(e)}", style="red")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="RAG Agent for Custom Data Analysis")
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start in interactive mode (default)'
    )
    
    parser.add_argument(
        '--add-docs',
        type=str,
        help='Add documents from specified path'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Process a single query and exit'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show knowledge base statistics and exit'
    )
    
    parser.add_argument(
        '--health',
        action='store_true',
        help='Show system health and exit'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Include subdirectories when adding documents'
    )
    
    args = parser.parse_args()
    
    cli = RAGCLI()
    
    try:
        if args.add_docs:
            cli.add_documents_batch(args.add_docs, recursive=args.recursive)
        elif args.query:
            cli._process_query(args.query)
        elif args.stats:
            cli._show_stats()
        elif args.health:
            cli._show_health()
        else:
            # Default to interactive mode
            cli.interactive_mode()
            
    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style="green")
    except Exception as e:
        console.print(f"\n❌ Fatal error: {str(e)}", style="red")
        logger.error(f"CLI fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
