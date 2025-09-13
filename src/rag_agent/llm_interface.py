"""
Ollama LLM interface for the RAG Agent system.
Handles communication with local Ollama models.
"""

import logging
import json
from typing import Dict, Any, Optional, List, AsyncGenerator
import requests
from langchain_ollama import OllamaLLM as LangChainOllama
from langchain.callbacks.manager import CallbackManagerForLLMRun
from .config import OllamaConfig

logger = logging.getLogger(__name__)

class OllamaLLM:
    """Interface for Ollama LLM integration."""
    
    def __init__(self, config: OllamaConfig):
        """Initialize Ollama LLM interface.
        
        Args:
            config: Ollama configuration object
        """
        self.config = config
        self.langchain_llm = LangChainOllama(
            base_url=config.base_url,
            model=config.model_name,
            temperature=config.temperature,
            timeout=config.timeout
        )
        self._validate_connection()
    
    def _validate_connection(self) -> None:
        """Validate connection to Ollama server."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {self.config.base_url}")
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if not any(name.startswith(self.config.model_name) for name in model_names):
                logger.warning(f"Model {self.config.model_name} not found. Available models: {model_names}")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama server: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from the LLM.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated text response
        """
        try:
            # Override default parameters with kwargs
            generation_params = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.context_window),
            }
            
            response = self.langchain_llm.invoke(prompt, **generation_params)
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate response from the LLM.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated text response
        """
        try:
            response = await self.langchain_llm.ainvoke(prompt, **kwargs)
            return response
        except Exception as e:
            logger.error(f"Error in async generation: {e}")
            raise
    
    def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response generation from the LLM.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional parameters for generation
            
        Yields:
            Chunks of generated text
        """
        try:
            for chunk in self.langchain_llm.stream(prompt, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        try:
            response = requests.get(f"{self.config.base_url}/api/show", 
                                  json={"name": self.config.model_name},
                                  timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get model info: {response.status_code}"}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models on the Ollama server.
        
        Returns:
            List of dictionaries containing model information
        """
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json().get("models", [])
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.config.base_url}/api/pull",
                json={"name": model_name},
                timeout=600  # Model pulling can take a long time
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def create_rag_prompt(self, query: str, context: str, system_message: Optional[str] = None) -> str:
        """Create a properly formatted prompt for RAG.
        
        Args:
            query: User query
            context: Retrieved context from documents
            system_message: Optional system message
            
        Returns:
            Formatted prompt string
        """
        if system_message is None:
            system_message = """You are a helpful AI assistant that answers questions based on the provided context. 
            Use only the information from the context to answer the question. 
            If the context doesn't contain enough information to answer the question, say so.
            Always cite the sources when possible."""
        
        prompt = f"""System: {system_message}

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def validate_response(self, response: str) -> bool:
        """Validate if the generated response is appropriate.
        
        Args:
            response: Generated response to validate
            
        Returns:
            True if response is valid, False otherwise
        """
        # Basic validation checks
        if not response or not response.strip():
            return False
        
        # Check for common error patterns
        error_patterns = [
            "I don't have access",
            "I cannot",
            "I'm not able to",
            "Error:",
            "Exception:"
        ]
        
        response_lower = response.lower()
        if any(pattern.lower() in response_lower for pattern in error_patterns):
            return False
        
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation: approximately 4 characters per token
        return len(text) // 4
    
    def truncate_to_context_window(self, text: str) -> str:
        """Truncate text to fit within the model's context window.
        
        Args:
            text: Input text to truncate
            
        Returns:
            Truncated text that fits within context window
        """
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= self.config.context_window:
            return text
        
        # Truncate to 90% of context window to leave room for response
        max_chars = int(self.config.context_window * 0.9 * 4)
        return text[:max_chars] + "..."

class OllamaModelManager:
    """Manager for Ollama model operations."""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
    
    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available, pull if necessary.
        
        Args:
            model_name: Name of the model to ensure is available
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Check if model is already available
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                if any(name.startswith(model_name) for name in available_models):
                    return True
            
            # Model not available, try to pull it
            logger.info(f"Pulling model {model_name}...")
            pull_response = requests.post(
                f"{self.config.base_url}/api/pull",
                json={"name": model_name},
                timeout=1800  # 30 minutes timeout for model pulling
            )
            
            return pull_response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error ensuring model availability: {e}")
            return False
    
    def get_recommended_model(self, use_case: str) -> str:
        """Get recommended model based on use case.
        
        Args:
            use_case: The intended use case
            
        Returns:
            Recommended model name
        """
        recommendations = {
            "general": "llama3",
            "code": "codellama",
            "technical": "llama3",
            "research": "llama3:70b",
            "business": "mistral",
            "resource_constrained": "mistral:7b"
        }
        
        return recommendations.get(use_case.lower(), "llama3")