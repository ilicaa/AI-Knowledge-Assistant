"""
LLM Client Module

Handles communication with LLM APIs (OpenAI and Groq).
Provides unified interface for generating responses with proper error handling.

Key features:
- Support for multiple providers (OpenAI, Groq)
- Retry logic with exponential backoff
- Timeout handling
- Rate limit awareness
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from app.config import (
    OPENAI_API_KEY,
    GROQ_API_KEY,
    LLM_PROVIDER,
    OPENAI_MODEL,
    GROQ_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    REQUEST_TIMEOUT,
)


@dataclass
class LLMResponse:
    """Represents a response from the LLM."""
    content: str
    model: str
    provider: str
    tokens_used: int
    latency_seconds: float
    success: bool
    error_message: Optional[str] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the client is properly configured."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self._client = None
        
        if self.is_configured():
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=REQUEST_TIMEOUT
            )
            logger.info(f"OpenAI client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self._client = None
    
    def is_configured(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.api_key and self.api_key != "your_openai_api_key_here")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response using OpenAI API."""
        if not self._client:
            return LLMResponse(
                content="",
                model=self.model,
                provider="openai",
                tokens_used=0,
                latency_seconds=0,
                success=False,
                error_message="OpenAI client not initialized. Check API key."
            )
        
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            
            logger.debug(f"OpenAI response generated in {latency:.2f}s ({tokens} tokens)")
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider="openai",
                tokens_used=tokens,
                latency_seconds=latency,
                success=True
            )
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = str(e)
            
            # Classify error
            if "rate_limit" in error_msg.lower():
                logger.warning(f"OpenAI rate limit hit: {e}")
                error_msg = "Rate limit exceeded. Please try again in a moment."
            elif "timeout" in error_msg.lower():
                logger.warning(f"OpenAI timeout: {e}")
                error_msg = "Request timed out. Please try again."
            else:
                logger.error(f"OpenAI error: {e}")
            
            return LLMResponse(
                content="",
                model=self.model,
                provider="openai",
                tokens_used=0,
                latency_seconds=latency,
                success=False,
                error_message=error_msg
            )


class GroqClient(BaseLLMClient):
    """Groq API client - faster inference for meeting <5s requirement."""
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        self._client = None
        
        if self.is_configured():
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Groq client."""
        try:
            from groq import Groq
            self._client = Groq(
                api_key=self.api_key,
                timeout=REQUEST_TIMEOUT
            )
            logger.info(f"Groq client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self._client = None
    
    def is_configured(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.api_key and self.api_key != "your_groq_api_key_here")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response using Groq API."""
        if not self._client:
            return LLMResponse(
                content="",
                model=self.model,
                provider="groq",
                tokens_used=0,
                latency_seconds=0,
                success=False,
                error_message="Groq client not initialized. Check API key."
            )
        
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            
            logger.debug(f"Groq response generated in {latency:.2f}s ({tokens} tokens)")
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider="groq",
                tokens_used=tokens,
                latency_seconds=latency,
                success=True
            )
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = str(e)
            
            if "rate_limit" in error_msg.lower():
                logger.warning(f"Groq rate limit hit: {e}")
                error_msg = "Rate limit exceeded. Please try again in a moment."
            elif "timeout" in error_msg.lower():
                logger.warning(f"Groq timeout: {e}")
                error_msg = "Request timed out. Please try again."
            else:
                logger.error(f"Groq error: {e}")
            
            return LLMResponse(
                content="",
                model=self.model,
                provider="groq",
                tokens_used=0,
                latency_seconds=latency,
                success=False,
                error_message=error_msg
            )


class LLMClient:
    """
    Unified LLM client that supports multiple providers.
    
    Automatically selects the configured provider and provides fallback.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider to use ("openai" or "groq")
        """
        self.provider = provider or LLM_PROVIDER
        self._openai_client = OpenAIClient()
        self._groq_client = GroqClient()
        
        # Determine active client
        self._active_client = self._select_client()
        
        if self._active_client is None:
            logger.error("No LLM provider configured! Please set API keys in .env file.")
    
    def _select_client(self) -> Optional[BaseLLMClient]:
        """Select the appropriate client based on configuration."""
        if self.provider == "groq" and self._groq_client.is_configured():
            logger.info("Using Groq as LLM provider")
            return self._groq_client
        elif self.provider == "openai" and self._openai_client.is_configured():
            logger.info("Using OpenAI as LLM provider")
            return self._openai_client
        # Fallback to any configured provider
        elif self._groq_client.is_configured():
            logger.info("Falling back to Groq as LLM provider")
            return self._groq_client
        elif self._openai_client.is_configured():
            logger.info("Falling back to OpenAI as LLM provider")
            return self._openai_client
        return None
    
    def is_configured(self) -> bool:
        """Check if any LLM provider is configured."""
        return self._active_client is not None
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the active provider."""
        if self._active_client is None:
            return {"configured": False, "provider": None, "model": None}
        
        if isinstance(self._active_client, GroqClient):
            return {
                "configured": True,
                "provider": "groq",
                "model": GROQ_MODEL
            }
        else:
            return {
                "configured": True,
                "provider": "openai", 
                "model": OPENAI_MODEL
            }
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            
        Returns:
            LLMResponse object with the generated content
        """
        if not self._active_client:
            return LLMResponse(
                content="",
                model="none",
                provider="none",
                tokens_used=0,
                latency_seconds=0,
                success=False,
                error_message="No LLM provider configured. Please set OPENAI_API_KEY or GROQ_API_KEY in your .env file."
            )
        
        return self._active_client.generate(prompt, system_prompt)


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def generate_response(prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
    """Convenience function to generate an LLM response."""
    client = get_llm_client()
    return client.generate(prompt, system_prompt)
