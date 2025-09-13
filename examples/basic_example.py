"""
Example script demonstrating basic RAG Agent usage.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent import RAGAgent, RAGConfig

def main():
    """Demonstrate basic RAG Agent functionality."""
    
    print("RAG Agent Basic Example")
    print("=" * 50)
    
    # Initialize RAG Agent
    print("1. Initializing RAG Agent...")
    agent = RAGAgent()
    
    if not agent.initialize():
        print("Failed to initialize RAG Agent!")
        return
    
    print("✓ RAG Agent initialized successfully!")
    
    # Ingest example documents
    print("\n2. Ingesting example documents...")
    examples_dir = Path(__file__).parent / "documents"
    
    if examples_dir.exists():
        stats = agent.ingest_directory(examples_dir)
        if stats['success']:
            print(f"✓ Ingested {stats['total_documents']} document chunks")
        else:
            print(f"Failed to ingest documents: {stats.get('error')}")
            return
    else:
        print("No example documents found. Creating sample text...")
        sample_text = """
        Artificial Intelligence (AI) is a broad field that encompasses machine learning, 
        natural language processing, computer vision, and robotics. Machine learning is 
        a subset of AI that focuses on algorithms that can learn from data. Deep learning 
        is a subset of machine learning that uses neural networks with multiple layers.
        
        Large Language Models (LLMs) are a type of AI model trained on vast amounts of 
        text data. They can understand and generate human-like text. Examples include 
        GPT, BERT, and LLaMA models.
        """
        
        if agent.ingest_text(sample_text, {"source": "sample_text", "topic": "AI"}):
            print("✓ Sample text ingested successfully!")
        else:
            print("Failed to ingest sample text!")
            return
    
    # Example queries
    print("\n3. Testing queries...")
    
    queries = [
        "What is artificial intelligence?",
        "What are the benefits of RAG?",
        "How does machine learning work?",
        "What are large language models?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        response = agent.query(query, include_sources=True)
        
        print(f"Answer: {response.answer}")
        print(f"Sources: {response.num_sources_used}")
        print(f"Time: {response.response_time:.3f}s")
        
        if response.sources:
            print("Source files:")
            for source in response.sources:
                file_name = source['metadata'].get('file_name', 'Unknown')
                score = source['similarity_score']
                print(f"  - {file_name} (Score: {score:.3f})")
    
    # Show system statistics
    print("\n4. System Statistics:")
    print("-" * 40)
    stats = agent.get_system_stats()
    print(f"Documents in database: {stats.get('vector_store', {}).get('total_documents', 0)}")
    print(f"Total conversations: {stats.get('conversation_count', 0)}")
    
    if 'retrieval' in stats:
        ret_stats = stats['retrieval']
        print(f"Average retrieval time: {ret_stats.get('avg_retrieval_time', 0):.3f}s")
    
    print("\n✓ Example completed successfully!")

if __name__ == "__main__":
    main()