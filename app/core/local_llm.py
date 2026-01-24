"""
Local LLM integration for LM Studio, Ollama, and other local LLM servers.
"""
import requests
import json
from typing import Optional, Dict, List
import socket


class LocalLLMClient:
    """Client for connecting to local LLM servers."""
    
    def __init__(self, provider: str = "auto", base_url: str = None, 
                 ollama_url: str = None, api_key: str = None, model_name: str = None, log_fn=None):
        """
        Initialize local LLM client.
        
        Args:
            provider: Provider type (auto, lm_studio, ollama, custom)
            base_url: Base URL for LM Studio or custom API
            ollama_url: Base URL for Ollama
            api_key: Optional API key
            model_name: Model name for Ollama
            log_fn: Optional logging function
        """
        self.provider = provider
        self.base_url = base_url or "http://localhost:1234/v1"
        self.ollama_url = ollama_url or "http://localhost:11434"
        self.api_key = api_key
        self.model_name = model_name
        self.log_fn = log_fn
        self._detected_provider = None
        
    def _log(self, msg: str):
        """Log a message if log function is available."""
        if self.log_fn:
            self.log_fn(msg)
    
    def _check_port(self, host: str, port: int) -> bool:
        """Check if a port is open."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    def detect_provider(self) -> Optional[str]:
        """
        Automatically detect which local LLM provider is running.
        
        Returns:
            Provider name or None if none detected
        """
        if self.provider != "auto":
            return self.provider
        
        # Check LM Studio (default port 1234)
        if self._check_port("localhost", 1234):
            self._detected_provider = "lm_studio"
            self._log("Detected LM Studio on port 1234")
            return "lm_studio"
        
        # Check Ollama (default port 11434)
        if self._check_port("localhost", 11434):
            self._detected_provider = "ollama"
            self._log("Detected Ollama on port 11434")
            return "ollama"
        
        # Check common alternative ports
        for port in [8000, 8080, 5000, 7860]:
            if self._check_port("localhost", port):
                self._detected_provider = "custom"
                self._log(f"Detected LLM server on port {port}")
                return "custom"
        
        return None
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from the local LLM server.
        
        Returns:
            List of model names
        """
        provider = self.detect_provider() if self.provider == "auto" else self.provider
        
        if provider == "lm_studio":
            try:
                response = requests.get(f"{self.base_url}/models", timeout=5)
                if response.status_code == 200:
                    models = response.json()
                    if isinstance(models, dict) and "data" in models:
                        return [m.get("id", "") for m in models["data"]]
                    return [m.get("id", "") for m in models] if isinstance(models, list) else []
            except Exception as e:
                self._log(f"Error fetching LM Studio models: {e}")
        
        elif provider == "ollama":
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return [m.get("name", "") for m in data.get("models", [])]
            except Exception as e:
                self._log(f"Error fetching Ollama models: {e}")
        
        return []
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Optional[str]:
        """
        Generate text using the local LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text or None if error
        """
        provider = self.detect_provider() if self.provider == "auto" else self.provider
        
        if not provider:
            self._log("No local LLM provider detected")
            return None
        
        try:
            if provider == "lm_studio":
                return self._generate_lm_studio(prompt, max_tokens, temperature)
            elif provider == "ollama":
                return self._generate_ollama(prompt, max_tokens, temperature)
            elif provider == "custom":
                return self._generate_custom(prompt, max_tokens, temperature)
        except Exception as e:
            self._log(f"Error generating text: {e}")
            return None
    
    def _generate_lm_studio(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Generate using LM Studio API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Get first available model if not specified
        models = self.get_available_models()
        model = models[0] if models else "local-model"
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            self._log(f"LM Studio API error: {response.status_code}")
            return None
    
    def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Generate using Ollama API."""
        model = self.model_name or "llama2"  # Default model
        
        data = {
            "model": model,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            self._log(f"Ollama API error: {response.status_code}")
            return None
    
    def _generate_custom(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Generate using custom API (OpenAI-compatible)."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {
            "model": self.model_name or "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            self._log(f"Custom API error: {response.status_code}")
            return None


def get_local_llm_client(provider: str = "auto", base_url: str = None, 
                         ollama_url: str = None, api_key: str = None, 
                         model_name: str = None, log_fn=None) -> LocalLLMClient:
    """
    Get a local LLM client instance.
    
    Args:
        provider: Provider type
        base_url: Base URL for API
        ollama_url: Ollama URL
        api_key: API key
        model_name: Model name
        log_fn: Logging function
        
    Returns:
        LocalLLMClient instance
    """
    return LocalLLMClient(provider, base_url, ollama_url, api_key, model_name, log_fn)
