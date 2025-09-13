#!/usr/bin/env python3
"""
Main entry point for the RAG Agent for Custom Data Analysis
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Main entry point with command routing"""
    parser = argparse.ArgumentParser(
        description="RAG Agent for Custom Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py cli                          # Start CLI interface
  python main.py web                          # Start web interface  
  python main.py add /path/to/docs            # Add documents
  python main.py query "What are the trends?" # Ask a question
  python main.py test                         # Run system tests
  python main.py health                       # Check system health
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # CLI command
    cli_parser = subparsers.add_parser('cli', help='Start CLI interface')
    cli_parser.add_argument('--interactive', '-i', action='store_true', 
                           help='Start in interactive mode (default)')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--port', type=int, default=8501, 
                           help='Port for web interface (default: 8501)')
    
    # Add documents command
    add_parser = subparsers.add_parser('add', help='Add documents to knowledge base')
    add_parser.add_argument('path', help='Path to documents (file or directory)')
    add_parser.add_argument('--recursive', '-r', action='store_true',
                           help='Include subdirectories')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Ask a question')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--type', choices=['auto', 'factual', 'analytical', 'comparative', 'summary'],
                             default='auto', help='Query type')
    query_parser.add_argument('--method', choices=['semantic', 'hybrid'],
                             default='semantic', help='Retrieval method')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check system health')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show knowledge base statistics')
    
    args = parser.parse_args()
    
    # If no command specified, default to CLI
    if not args.command:
        args.command = 'cli'
        args.interactive = True
    
    # Route to appropriate handler
    if args.command == 'cli':
        from interfaces.cli import RAGCLI
        cli = RAGCLI()
        cli.interactive_mode()
        
    elif args.command == 'web':
        import subprocess
        cmd = ['streamlit', 'run', 'interfaces/web_app.py', '--server.port', str(args.port)]
        subprocess.run(cmd)
        
    elif args.command == 'add':
        from src.core.rag_agent import rag_agent
        try:
            result = rag_agent.add_documents(args.path, recursive=args.recursive)
            print(f"✅ Added {result['documents_added']} documents successfully")
            print(f"   Total in knowledge base: {result['total_documents_in_kb']}")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            sys.exit(1)
            
    elif args.command == 'query':
        from src.core.rag_agent import rag_agent
        try:
            response = rag_agent.query(
                question=args.question,
                query_type=args.type,
                retrieval_method=args.method
            )
            print(f"\n🤖 Answer:")
            print(f"{response.answer}")
            print(f"\n📊 Stats:")
            print(f"   Confidence: {response.confidence_score:.1%}")
            print(f"   Sources: {len(response.sources)}")
            print(f"   Time: {response.processing_time:.2f}s")
            
            if response.sources:
                print(f"\n📚 Sources:")
                for source in response.sources:
                    print(f"   - {source['filename']} (relevance: {source['relevance_score']:.3f})")
                    
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            sys.exit(1)
            
    elif args.command == 'test':
        import subprocess
        result = subprocess.run([sys.executable, 'test_installation.py'])
        sys.exit(result.returncode)
        
    elif args.command == 'health':
        from src.core.rag_agent import rag_agent
        try:
            health = rag_agent.health_check()
            print(f"🏥 System Health: {health['status'].upper()}")
            print("\nComponents:")
            for component, details in health.get('components', {}).items():
                status_icon = "✅" if details['status'] == 'healthy' else "❌"
                print(f"   {status_icon} {component.replace('_', ' ').title()}: {details['status']}")
                
        except Exception as e:
            print(f"❌ Error checking health: {e}")
            sys.exit(1)
            
    elif args.command == 'stats':
        from src.core.rag_agent import rag_agent
        try:
            stats = rag_agent.get_knowledge_base_stats()
            kb_stats = stats.get('knowledge_base', {})
            
            print("📊 Knowledge Base Statistics")
            print(f"   Total Documents: {kb_stats.get('total_documents', 0)}")
            print(f"   Unique Sources: {kb_stats.get('unique_sources', 0)}")
            
            file_types = kb_stats.get('file_types', {})
            if file_types:
                print("   File Types:")
                for file_type, count in file_types.items():
                    print(f"     {file_type}: {count}")
                    
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
