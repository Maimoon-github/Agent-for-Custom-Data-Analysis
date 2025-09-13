"""
Test script to verify the RAG Agent installation and functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.core.rag_agent import rag_agent
        print("✓ RAG Agent import successful")
    except Exception as e:
        print(f"✗ RAG Agent import failed: {e}")
        return False
    
    try:
        from src.core.document_processor import document_processor
        print("✓ Document Processor import successful")
    except Exception as e:
        print(f"✗ Document Processor import failed: {e}")
        return False
    
    try:
        from src.core.vector_database import vector_db
        print("✓ Vector Database import successful")
    except Exception as e:
        print(f"✗ Vector Database import failed: {e}")
        return False
    
    return True

def test_system_health():
    """Test system health"""
    print("\nTesting system health...")
    
    try:
        from src.core.rag_agent import rag_agent
        health = rag_agent.health_check()
        
        print(f"System Status: {health['status']}")
        
        for component, details in health.get('components', {}).items():
            status = details['status']
            print(f"  {component}: {status}")
        
        return health['status'] in ['healthy', 'degraded']
        
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_sample_document():
    """Test with a sample document"""
    print("\nTesting with sample document...")
    
    # Create a sample document
    sample_content = """
# Sample Research Paper

## Abstract
This is a sample research paper for testing the RAG Agent system. The paper discusses the importance of local AI systems for data privacy and security.

## Introduction
Artificial Intelligence has become increasingly important in data analysis. However, many AI systems require sending data to external servers, which raises privacy concerns.

## Methodology
We propose a local RAG (Retrieval-Augmented Generation) system that processes all data locally without external API calls.

## Results
Our testing shows that local RAG systems can provide accurate results while maintaining complete data privacy.

## Conclusion
Local AI systems represent the future of privacy-preserving data analysis.
"""
    
    # Save sample document
    sample_dir = Path("data/documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_file = sample_dir / "sample_paper.txt"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    try:
        from src.core.rag_agent import rag_agent
        
        # Add the sample document
        print("Adding sample document...")
        result = rag_agent.add_documents(str(sample_file))
        
        if result['status'] == 'success':
            print(f"✓ Added {result['documents_added']} document chunks")
        else:
            print(f"✗ Failed to add document: {result}")
            return False
        
        # Test a query
        print("Testing query...")
        response = rag_agent.query("What is the main topic of this research?")
        
        print(f"Query: What is the main topic of this research?")
        print(f"Answer: {response.answer[:200]}...")
        print(f"Confidence: {response.confidence_score:.1%}")
        print(f"Sources: {len(response.sources)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample document test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("RAG Agent Installation Test")
    print("=" * 40)
    
    tests = [
        ("Import Tests", test_imports),
        ("System Health", test_system_health),
        ("Sample Document", test_sample_document)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        if test_func():
            print(f"✓ {test_name} PASSED")
            passed += 1
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG Agent is ready to use.")
        print("\nNext steps:")
        print("1. Add your documents: python interfaces/cli.py --add-docs /path/to/docs")
        print("2. Start the CLI: python interfaces/cli.py")
        print("3. Start the web app: streamlit run interfaces/web_app.py")
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
