#!/usr/bin/env python3
"""
Multi-LLM handler supporting OpenAI GPT, Ollama, and other providers.
Unified interface for querying different LLM models.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.llm.model import LLM_response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    name: str
    provider: str  # 'openai', 'ollama', 'anthropic'
    model: str
    max_tokens: int = 1000
    temperature: float = 0.1
    retry_attempts: int = 3
    retry_delay: float = 1.0


class MultiLLMHandler:
    """
    Unified handler for querying multiple LLM providers
    """
    
    def __init__(self):
        # Define available models
        self.models = self._initialize_models()
        
    def _initialize_models(self) -> Dict[str, LLMConfig]:
        """Initialize model configurations"""
        models = {}
        
        # OpenAI models
        models['gpt-4o-mini'] = LLMConfig(
            name='gpt-4o-mini',
            provider='openai',
            model='gpt-4o-mini',
            max_tokens=1000,
            temperature=0.1
        )
        
        models['gpt-3.5-turbo'] = LLMConfig(
            name='gpt-3.5-turbo',
            provider='openai',
            model='gpt-3.5-turbo-0125',
            max_tokens=1000,
            temperature=0.1
        )
        
        models['gpt-4-turbo'] = LLMConfig(
            name='gpt-4-turbo',
            provider='openai',
            model='gpt-4-turbo',
            max_tokens=1000,
            temperature=0.1
        )
        
        # Ollama models (require local Ollama installation)
        ollama_models = [
            'llama3',
            'llama3.2',
            'mistral',
            'gemma:2b'
        ]
        
        for model in ollama_models:
            models[model] = LLMConfig(
                name=model,
                provider='ollama',
                model=model,
                max_tokens=1000,
                temperature=0.1
            )
        
        return models
    
    def query_model(self, 
                   prompt: str, 
                   model_name: str,
                   verbose: bool = False) -> Dict[str, Any]:
        """
        Query a single LLM model with the prompt
        
        Args:
            prompt: The prompt to send
            model_name: Name of the model to use
            verbose: Whether to print progress
            
        Returns:
            Dict with 'response' and 'model' keys
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        config = self.models[model_name]
        
        if verbose:
            logger.info(f"Querying {model_name} ({config.provider})...")
        
        start_time = time.time()
        
        # Query with retry logic
        for attempt in range(config.retry_attempts):
            try:
                if config.provider == 'openai':
                    response, history = LLM_response(
                        chat_prompt=prompt,
                        llm_model=config.model,
                        history=[],
                        kwargs={},
                        max_tokens=config.max_tokens
                    )
                elif config.provider == 'ollama':
                    response, history = LLM_response(
                        chat_prompt=prompt,
                        llm_model=config.model,
                        history=[],
                        kwargs={},
                        max_tokens=config.max_tokens
                    )
                else:
                    raise ValueError(f"Unsupported provider: {config.provider}")
                
                elapsed = time.time() - start_time
                
                if verbose:
                    logger.info(f"✓ {model_name} completed in {elapsed:.2f}s")
                
                return {
                    'response': response,
                    'model': model_name,
                    'provider': config.provider,
                    'elapsed_time': elapsed,
                    'status': 'success'
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{config.retry_attempts} failed for {model_name}: {e}")
                if attempt < config.retry_attempts - 1:
                    time.sleep(config.retry_delay * (attempt + 1))
                else:
                    elapsed = time.time() - start_time
                    return {
                        'response': None,
                        'model': model_name,
                        'provider': config.provider,
                        'elapsed_time': elapsed,
                        'status': 'error',
                        'error': str(e)
                    }
    
    def query_batch(self,
                    prompt: str,
                    model_names: List[str],
                    verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Query multiple models with the same prompt
        
        Args:
            prompt: The prompt to send
            model_names: List of model names to query
            verbose: Whether to print progress
            
        Returns:
            Dict mapping model_name -> response dict
        """
        results = {}
        
        if verbose:
            logger.info(f"Querying {len(model_names)} models: {', '.join(model_names)}")
        
        for model_name in model_names:
            results[model_name] = self.query_model(prompt, model_name, verbose=verbose)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def get_models_by_family(self) -> Dict[str, List[str]]:
        """Group models by provider/family"""
        families = {}
        for name, config in self.models.items():
            provider = config.provider
            if provider not in families:
                families[provider] = []
            families[provider].append(name)
        return families


def test_multi_llm():
    """Test the multi-LLM handler"""
    handler = MultiLLMHandler()
    
    print("Available models:")
    for provider, models in handler.get_models_by_family().items():
        print(f"  {provider}: {', '.join(models)}")
    
    # Simple test prompt
    test_prompt = """You are evaluating whether two data tables are semantically related.

Table A:
col1,col2,col3
value1,value2,value3

Table B:
col1,col2
value1,value2

Task: Determine if Tables A and B are related (YES/NO/UNSURE) and explain why in 1-2 sentences.

Respond in YAML format:
related: [YES/NO/UNSURE]
rationale: "[your explanation]" """
    
    # Test with a single model
    print("\nTesting with gpt-4o-mini...")
    result = handler.query_model(test_prompt, 'gpt-4o-mini', verbose=True)
    print(f"Response: {result.get('response', 'None')[:200]}")
    
    # Print available models
    print(f"\n✓ Available models: {handler.get_available_models()}")


if __name__ == "__main__":
    test_multi_llm()

