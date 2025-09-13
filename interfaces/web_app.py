"""
Streamlit Web Interface for the RAG Agent
Provides a user-friendly web interface for document analysis
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.rag_agent import rag_agent

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Agent - Custom Data Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = rag_agent
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_system():
    """Initialize the RAG system and check health"""
    try:
        health = st.session_state.agent.health_check()
        stats = st.session_state.agent.get_knowledge_base_stats()
        
        st.session_state.health = health
        st.session_state.stats = stats
        st.session_state.initialized = True
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False

def display_sidebar():
    """Display the sidebar with system information and controls"""
    with st.sidebar:
        st.title("🤖 RAG Agent")
        st.markdown("---")
        
        # System Health
        if st.session_state.initialized:
            health = st.session_state.health
            if health["status"] == "healthy":
                st.success("✅ System Healthy")
            elif health["status"] == "degraded":
                st.warning("⚠️ System Degraded")
            else:
                st.error("❌ System Unhealthy")
            
            # Knowledge Base Stats
            kb_stats = st.session_state.stats.get("knowledge_base", {})
            doc_count = kb_stats.get("total_documents", 0)
            
            st.metric("📚 Documents", doc_count)
            
            if kb_stats.get("file_types"):
                st.write("**File Types:**")
                for file_type, count in kb_stats["file_types"].items():
                    st.write(f"  {file_type}: {count}")
        
        st.markdown("---")
        
        # Document Management
        st.subheader("📂 Document Management")
        
        if st.button("🔄 Refresh Stats"):
            st.session_state.stats = st.session_state.agent.get_knowledge_base_stats()
            st.rerun()
        
        if st.button("🗑️ Reset Knowledge Base"):
            if st.confirm("Are you sure? This will delete all documents."):
                if st.session_state.agent.reset_knowledge_base():
                    st.success("Knowledge base reset successfully")
                    st.session_state.stats = st.session_state.agent.get_knowledge_base_stats()
                    st.rerun()
                else:
                    st.error("Failed to reset knowledge base")
        
        st.markdown("---")
        
        # Query History
        st.subheader("📋 Query History")
        if st.session_state.query_history:
            for i, query in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                with st.expander(f"Query {len(st.session_state.query_history) - i + 1}"):
                    st.write(f"**Q:** {query['question'][:100]}...")
                    st.write(f"**Confidence:** {query['confidence']:.1%}")
                    st.write(f"**Time:** {query['processing_time']:.2f}s")
        else:
            st.write("No queries yet")

def display_main_interface():
    """Display the main query interface"""
    st.title("🤖 RAG Agent for Custom Data Analysis")
    st.markdown("Ask questions about your documents and get AI-powered insights")
    
    # Initialize system if not done
    if not st.session_state.initialized:
        with st.spinner("Initializing RAG system..."):
            if not initialize_system():
                st.stop()
    
    # Check if knowledge base is empty
    kb_stats = st.session_state.stats.get("knowledge_base", {})
    doc_count = kb_stats.get("total_documents", 0)
    
    if doc_count == 0:
        st.warning("⚠️ No documents in knowledge base. Please add documents first.")
        st.stop()
    
    # Query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main trends in the sales data?",
            key="user_question"
        )
    
    with col2:
        query_type = st.selectbox(
            "Query Type:",
            ["auto", "factual", "analytical", "comparative", "summary"],
            help="Auto-detect will automatically determine the best approach"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            retrieval_method = st.selectbox(
                "Retrieval Method:",
                ["semantic", "hybrid"],
                help="Semantic uses vector similarity, hybrid combines with keyword search"
            )
        
        with col2:
            k_documents = st.slider(
                "Documents to retrieve:",
                1, 20, 5,
                help="Number of documents to consider for answering"
            )
        
        with col3:
            temperature = st.slider(
                "Response creativity:",
                0.0, 1.0, 0.7,
                help="Higher values = more creative, lower = more factual"
            )
    
    # Process query
    if user_question and st.button("🔍 Ask Question", type="primary"):
        process_query(user_question, query_type, retrieval_method, k_documents, temperature)

def process_query(question: str, query_type: str, retrieval_method: str, k: int, temperature: float):
    """Process a user query and display results"""
    with st.spinner("🤔 Thinking..."):
        try:
            start_time = time.time()
            
            # Get response from RAG agent
            response = st.session_state.agent.query(
                question=question,
                query_type=query_type,
                retrieval_method=retrieval_method,
                k=k,
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            
            # Display response
            display_response(response)
            
            # Add to history
            st.session_state.query_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': response.answer,
                'confidence': response.confidence_score,
                'sources_count': len(response.sources),
                'processing_time': processing_time,
                'metadata': response.metadata
            })
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")

def display_response(response):
    """Display the RAG response"""
    # Main answer
    confidence_color = "green" if response.confidence_score > 0.7 else "orange" if response.confidence_score > 0.4 else "red"
    
    st.markdown("### 💡 Answer")
    st.markdown(
        f'<div style="border-left: 4px solid {confidence_color}; padding-left: 1rem; background-color: rgba(255,255,255,0.05); border-radius: 0.5rem; padding: 1rem;">'
        f'{response.answer}'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Confidence and stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Confidence", f"{response.confidence_score:.1%}")
    
    with col2:
        st.metric("⏱️ Processing Time", f"{response.processing_time:.2f}s")
    
    with col3:
        st.metric("📄 Documents Found", response.retrieval_stats['documents_found'])
    
    with col4:
        st.metric("📚 Sources Used", len(response.sources))
    
    # Sources
    if response.sources:
        st.markdown("### 📚 Sources")
        
        # Create sources dataframe
        sources_data = []
        for source in response.sources:
            sources_data.append({
                'Rank': source['rank'],
                'Document': source['filename'],
                'Type': source.get('document_type', ''),
                'Relevance': f"{source['relevance_score']:.3f}",
                'Chunk': source.get('chunk_index', 0)
            })
        
        sources_df = pd.DataFrame(sources_data)
        st.dataframe(sources_df, use_container_width=True)
        
        # Sources visualization
        if len(response.sources) > 1:
            fig = px.bar(
                sources_df, 
                x='Document', 
                y='Relevance',
                title="Source Relevance Scores",
                hover_data=['Type', 'Chunk']
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Advanced details
    with st.expander("🔍 Advanced Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Retrieval Information:**")
            st.json({
                "method": response.metadata.get("retrieval_method", "unknown"),
                "query_type": response.metadata.get("query_type", "unknown"),
                "documents_found": response.retrieval_stats.get("documents_found", 0),
                "retrieval_time": f"{response.retrieval_stats.get('retrieval_time', 0):.3f}s",
                "average_score": f"{response.retrieval_stats.get('average_score', 0):.3f}"
            })
        
        with col2:
            st.write("**Generation Information:**")
            st.json({
                "model": response.generation_stats.get("model", "unknown"),
                "generation_time": f"{response.generation_stats.get('generation_time', 0):.3f}s",
                "input_tokens": response.generation_stats.get("token_usage", {}).get("input_tokens", 0),
                "output_tokens": response.generation_stats.get("token_usage", {}).get("output_tokens", 0)
            })

def display_document_management():
    """Display document management interface"""
    st.header("📂 Document Management")
    
    # File upload
    st.subheader("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'csv'],
        help="Supported formats: PDF, TXT, DOCX, CSV"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Uploaded Files"):
            process_uploaded_files(uploaded_files)
    
    # Directory input
    st.subheader("📁 Add from Directory")
    
    directory_path = st.text_input(
        "Directory path:",
        placeholder="C:/Documents/MyData",
        help="Enter the full path to a directory containing documents"
    )
    
    recursive = st.checkbox("Include subdirectories", value=True)
    
    if directory_path and st.button("📂 Add Directory"):
        add_directory(directory_path, recursive)
    
    # Current knowledge base
    st.subheader("📊 Current Knowledge Base")
    
    if st.session_state.initialized:
        stats = st.session_state.stats.get("knowledge_base", {})
        
        if stats.get("total_documents", 0) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Documents", stats.get("total_documents", 0))
                st.metric("Unique Sources", stats.get("unique_sources", 0))
            
            with col2:
                # File type distribution
                file_types = stats.get("file_types", {})
                if file_types:
                    fig = px.pie(
                        values=list(file_types.values()),
                        names=list(file_types.keys()),
                        title="Document Types Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No documents in knowledge base yet.")

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        with st.spinner(f"Processing {len(uploaded_files)} files..."):
            saved_files = []
            
            # Save uploaded files temporarily
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                saved_files.append(str(file_path))
            
            # Process each file
            total_added = 0
            for file_path in saved_files:
                try:
                    result = st.session_state.agent.add_documents(file_path)
                    total_added += result.get("documents_added", 0)
                except Exception as e:
                    st.error(f"Error processing {Path(file_path).name}: {str(e)}")
            
            # Update stats
            st.session_state.stats = st.session_state.agent.get_knowledge_base_stats()
            
            st.success(f"✅ Successfully processed files! Added {total_added} document chunks.")
            
    finally:
        # Clean up temporary files
        for file_path in temp_dir.glob("*"):
            file_path.unlink()
        temp_dir.rmdir()

def add_directory(directory_path: str, recursive: bool):
    """Add documents from a directory"""
    try:
        with st.spinner(f"Processing directory: {directory_path}"):
            result = st.session_state.agent.add_documents(directory_path, recursive=recursive)
            
            if result["status"] == "success":
                st.success(
                    f"✅ Successfully added {result['documents_added']} documents "
                    f"({result['documents_processed']} processed, "
                    f"{result['documents_failed']} failed) "
                    f"in {result['processing_time']:.2f}s"
                )
                
                # Update stats
                st.session_state.stats = st.session_state.agent.get_knowledge_base_stats()
            else:
                st.warning(f"⚠️ {result.get('message', 'Unknown error')}")
                
    except Exception as e:
        st.error(f"❌ Error adding directory: {str(e)}")

def display_analytics():
    """Display analytics and insights about queries"""
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.query_history:
        st.info("No query history available. Ask some questions first!")
        return
    
    # Convert history to dataframe
    df = pd.DataFrame(st.session_state.query_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", len(df))
    
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col3:
        avg_time = df['processing_time'].mean()
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    with col4:
        avg_sources = df['sources_count'].mean()
        st.metric("Avg Sources Used", f"{avg_sources:.1f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution
        fig = px.histogram(
            df, 
            x='confidence', 
            title="Confidence Score Distribution",
            nbins=10
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Processing time over time
        fig = px.line(
            df, 
            x='timestamp', 
            y='processing_time',
            title="Response Time Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent queries table
    st.subheader("Recent Queries")
    display_df = df[['timestamp', 'question', 'confidence', 'sources_count', 'processing_time']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df['processing_time'] = display_df['processing_time'].apply(lambda x: f"{x:.2f}s")
    display_df = display_df.rename(columns={
        'timestamp': 'Time',
        'question': 'Question',
        'confidence': 'Confidence',
        'sources_count': 'Sources',
        'processing_time': 'Time'
    })
    
    st.dataframe(display_df.tail(10), use_container_width=True)

def main():
    """Main Streamlit application"""
    # Sidebar
    display_sidebar()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🤖 Chat", "📂 Documents", "📊 Analytics"])
    
    with tab1:
        display_main_interface()
    
    with tab2:
        display_document_management()
    
    with tab3:
        display_analytics()

if __name__ == "__main__":
    main()
