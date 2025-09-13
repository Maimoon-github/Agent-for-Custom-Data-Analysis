"""
Ollama LLM integration for the RAG system
Handles local language model communication and response generation
"""

import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

import requests
from langchain_ollama import OllamaLLM
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from config.settings import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OllamaManager:
    """Manages Ollama LLM connections and interactions"""
    
    def __init__(self):
        self.config = config.ollama
        self.llm = None
        self.model_info = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Ollama LLM connection"""
        try:
            # Check if Ollama is running
            if not self._check_ollama_connection():
                raise ConnectionError("Ollama server is not running or not accessible")
            
            # Check if model is available
            if not self._check_model_availability():
                logger.warning(f"Model {self.config.model} not found. Attempting to pull...")
                self._pull_model()
            
            # Initialize LangChain Ollama LLM
            self.llm = OllamaLLM(
                base_url=self.config.base_url,
                model=self.config.model,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens,
                num_ctx=self.config.context_window
            )
            
            # Get model information
            self.model_info = self._get_model_info()
            
            logger.info(f"Ollama LLM initialized successfully with model: {self.config.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
            raise
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection check failed: {str(e)}")
            return False
    
    def _check_model_availability(self) -> bool:
        """Check if the specified model is available"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                return self.config.model in available_models
            return False
        except Exception as e:
            logger.error(f"Model availability check failed: {str(e)}")
            return False
    
    def _pull_model(self) -> bool:
        """Pull the specified model from Ollama"""
        try:
            logger.info(f"Pulling model {self.config.model}...")
            
            response = requests.post(
                f"{self.config.base_url}/api/pull",
                json={"name": self.config.model},
                stream=True,
                timeout=300  # 5 minutes timeout for model pulling
            )
            
            if response.status_code == 200:
                # Stream the pull progress
                for line in response.iter_lines():
                    if line:
                        try:
                            progress = json.loads(line)
                            status = progress.get('status', '')
                            if 'pulling' in status.lower():
                                logger.info(f"Pulling: {status}")
                        except json.JSONDecodeError:
                            continue
                
                logger.info(f"Model {self.config.model} pulled successfully")
                return True
            else:
                logger.error(f"Failed to pull model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.post(
                f"{self.config.base_url}/api/show",
                json={"name": self.config.model},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Could not get model info: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def generate_response(self, prompt: str, system_message: str = None, **kwargs) -> Dict[str, Any]:
        """Generate a response using the Ollama LLM"""
        try:
            start_time = time.time()
            
            # Prepare the full prompt
            if system_message:
                full_prompt = f"System: {system_message}\n\nHuman: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt
            
            # Override default parameters with any provided kwargs
            generation_params = {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'num_predict': kwargs.get('max_tokens', self.config.max_tokens),
                'top_p': kwargs.get('top_p', 0.9),
                'top_k': kwargs.get('top_k', 40),
            }
            
            # Generate response
            response = self.llm.invoke(
                full_prompt,
                **generation_params
            )
            
            generation_time = time.time() - start_time
            
            # Calculate approximate token counts (rough estimation)
            input_tokens = len(full_prompt.split())
            output_tokens = len(response.split())
            
            result = {
                'response': response,
                'generation_time': generation_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'model': self.config.model,
                'parameters': generation_params,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Generated response in {generation_time:.2f}s ({output_tokens} tokens)")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_streaming_response(self, prompt: str, system_message: str = None, **kwargs):
        """Generate a streaming response using Ollama API directly"""
        try:
            # Prepare the request
            if system_message:
                full_prompt = f"System: {system_message}\n\nHuman: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt
            
            payload = {
                "model": self.config.model,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 40),
                }
            }
            
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if not chunk.get('done', False):
                                yield chunk.get('response', '')
                        except json.JSONDecodeError:
                            continue
            else:
                raise Exception(f"Streaming request failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama (if embedding model is available)"""
        try:
            embeddings = []
            
            for text in texts:
                response = requests.post(
                    f"{self.config.base_url}/api/embeddings",
                    json={
                        "model": "nomic-embed-text",  # Default embedding model
                        "prompt": text
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    embedding_data = response.json()
                    embeddings.append(embedding_data.get('embedding', []))
                else:
                    logger.warning(f"Failed to generate embedding for text: {text[:50]}...")
                    embeddings.append([])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the Ollama connection and model availability"""
        try:
            # Test basic connection
            connection_ok = self._check_ollama_connection()
            
            # Test model availability
            model_available = self._check_model_availability()
            
            # Test generation if model is available
            generation_ok = False
            generation_time = None
            
            if model_available:
                try:
                    start_time = time.time()
                    test_response = self.generate_response("Hello, this is a test.")
                    generation_time = time.time() - start_time
                    generation_ok = bool(test_response.get('response'))
                except:
                    generation_ok = False
            
            return {
                'connection_ok': connection_ok,
                'model_available': model_available,
                'generation_ok': generation_ok,
                'generation_time': generation_time,
                'model': self.config.model,
                'base_url': self.config.base_url,
                'model_info': self.model_info
            }
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return {
                'connection_ok': False,
                'model_available': False,
                'generation_ok': False,
                'error': str(e)
            }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models in Ollama"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            else:
                logger.error(f"Failed to list models: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            # Check if the model is available
            available_models = [model['name'] for model in self.list_available_models()]
            
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not available. Attempting to pull...")
                if not self._pull_model_by_name(model_name):
                    return False
            
            # Update configuration and reinitialize
            self.config.model = model_name
            self._initialize_llm()
            
            logger.info(f"Switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching model to {model_name}: {str(e)}")
            return False
    
    def _pull_model_by_name(self, model_name: str) -> bool:
        """Pull a specific model by name"""
        try:
            response = requests.post(
                f"{self.config.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False

# Create global Ollama manager instance
ollama_manager = OllamaManager()
