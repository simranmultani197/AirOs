"""
Multi-model LLM provider support for AgentCircuit.

This module provides a unified interface for different LLM providers
(OpenAI, Anthropic, Groq, local models) with automatic fallback.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import os
import json
import time

from .errors import ProviderError


class ProviderType(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ProviderType
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 30.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_estimate(self) -> float:
        """Estimate cost based on typical pricing."""
        # Rough estimates - actual pricing varies by model
        prompt_cost = self.prompt_tokens * 0.000003  # $3 per 1M tokens
        completion_cost = self.completion_tokens * 0.000015  # $15 per 1M tokens
        return prompt_cost + completion_cost


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._last_usage: Optional[TokenUsage] = None

    @property
    def last_usage(self) -> Optional[TokenUsage]:
        """Get token usage from last call."""
        return self._last_usage

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The prompt to complete

        Returns:
            The completion text

        Raises:
            ProviderError on failure
        """
        pass

    def __call__(self, prompt: str) -> str:
        """Allow provider to be used as a callable."""
        return self.complete(prompt)


class OpenAIProvider(LLMProvider):
    """OpenAI/OpenAI-compatible provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderError(
                    "OpenAI package not installed. Run: pip install openai",
                    provider="openai"
                )

            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ProviderError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable.",
                    provider="openai"
                )

            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        return self._client

    def complete(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params
            )

            # Track usage
            if response.usage:
                self._last_usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )

            return response.choices[0].message.content

        except Exception as e:
            raise ProviderError(
                f"OpenAI API error: {str(e)}",
                provider="openai",
                original_error=e
            )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ProviderError(
                    "Anthropic package not installed. Run: pip install anthropic",
                    provider="anthropic"
                )

            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ProviderError(
                    "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.",
                    provider="anthropic"
                )

            self._client = Anthropic(api_key=api_key)
        return self._client

    def complete(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.config.model_id,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **self.config.extra_params
            )

            # Track usage
            if hasattr(response, 'usage'):
                self._last_usage = TokenUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens
                )

            return response.content[0].text

        except Exception as e:
            raise ProviderError(
                f"Anthropic API error: {str(e)}",
                provider="anthropic",
                original_error=e
            )


class GroqProvider(LLMProvider):
    """Groq provider for fast inference."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ProviderError(
                    "Groq package not installed. Run: pip install groq",
                    provider="groq"
                )

            api_key = self.config.api_key or os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ProviderError(
                    "Groq API key not found. Set GROQ_API_KEY environment variable.",
                    provider="groq"
                )

            self._client = Groq(api_key=api_key)
        return self._client

    def complete(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params
            )

            # Track usage
            if response.usage:
                self._last_usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )

            return response.choices[0].message.content

        except Exception as e:
            raise ProviderError(
                f"Groq API error: {str(e)}",
                provider="groq",
                original_error=e
            )


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"

    def complete(self, prompt: str) -> str:
        try:
            import httpx
        except ImportError:
            raise ProviderError(
                "httpx package not installed. Run: pip install httpx",
                provider="ollama"
            )

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.config.model_id,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                            **self.config.extra_params
                        }
                    }
                )
                response.raise_for_status()

                data = response.json()

                # Track usage (Ollama provides this)
                if "eval_count" in data:
                    self._last_usage = TokenUsage(
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                        total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                    )

                return data["response"]

        except Exception as e:
            raise ProviderError(
                f"Ollama error: {str(e)}",
                provider="ollama",
                original_error=e
            )


class CustomProvider(LLMProvider):
    """Wrapper for custom callable."""

    def __init__(self, config: ModelConfig, callable_fn: Callable[[str], str]):
        super().__init__(config)
        self.callable_fn = callable_fn

    def complete(self, prompt: str) -> str:
        try:
            return self.callable_fn(prompt)
        except Exception as e:
            raise ProviderError(
                f"Custom provider error: {str(e)}",
                provider="custom",
                original_error=e
            )


class ProviderChain:
    """
    Chain of providers with automatic fallback.

    Tries providers in order until one succeeds.
    """

    def __init__(self, providers: Optional[List[LLMProvider]] = None):
        self.providers = providers or []
        self._last_provider: Optional[LLMProvider] = None
        self._total_usage = TokenUsage()

    def add(self, provider: LLMProvider) -> "ProviderChain":
        """Add a provider to the chain."""
        self.providers.append(provider)
        return self

    @property
    def last_usage(self) -> TokenUsage:
        """Get combined token usage."""
        return self._total_usage

    def complete(self, prompt: str) -> str:
        """
        Try providers in order until one succeeds.

        Args:
            prompt: The prompt to complete

        Returns:
            Completion text

        Raises:
            ProviderError if all providers fail
        """
        errors = []

        for provider in self.providers:
            try:
                result = provider.complete(prompt)
                self._last_provider = provider

                # Track cumulative usage
                if provider.last_usage:
                    self._total_usage.prompt_tokens += provider.last_usage.prompt_tokens
                    self._total_usage.completion_tokens += provider.last_usage.completion_tokens
                    self._total_usage.total_tokens += provider.last_usage.total_tokens

                return result

            except ProviderError as e:
                errors.append(f"{e.provider}: {str(e)}")
                continue

        raise ProviderError(
            f"All providers failed: {'; '.join(errors)}",
            provider="chain"
        )

    def __call__(self, prompt: str) -> str:
        """Allow chain to be used as a callable."""
        return self.complete(prompt)


# Factory functions for easy provider creation
def create_provider(
    provider_type: Union[str, ProviderType],
    model_id: str,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Create a provider instance.

    Args:
        provider_type: Type of provider (openai, anthropic, groq, ollama)
        model_id: Model identifier
        api_key: Optional API key
        **kwargs: Additional configuration

    Returns:
        Configured LLMProvider instance
    """
    if isinstance(provider_type, str):
        provider_type = ProviderType(provider_type.lower())

    config = ModelConfig(
        provider=provider_type,
        model_id=model_id,
        api_key=api_key,
        **kwargs
    )

    provider_map = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.GROQ: GroqProvider,
        ProviderType.OLLAMA: OllamaProvider,
    }

    provider_class = provider_map.get(provider_type)
    if not provider_class:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return provider_class(config)


def create_default_chain() -> ProviderChain:
    """
    Create a default provider chain with auto-detected providers.

    Checks environment variables and adds available providers.
    """
    chain = ProviderChain()

    # Try Groq first (fastest)
    if os.environ.get("GROQ_API_KEY"):
        chain.add(create_provider("groq", "llama-3.3-70b-versatile"))

    # Try OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        chain.add(create_provider("openai", "gpt-4o-mini"))

    # Try Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        chain.add(create_provider("anthropic", "claude-3-5-haiku-latest"))

    # Try local Ollama
    try:
        import httpx
        with httpx.Client(timeout=2.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    chain.add(create_provider("ollama", models[0]["name"]))
    except Exception:
        pass

    if not chain.providers:
        raise ProviderError(
            "No LLM providers available. Set an API key (GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY) or run Ollama locally.",
            provider="none"
        )

    return chain


# Pre-configured model shortcuts
MODELS = {
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4-turbo": ("openai", "gpt-4-turbo"),
    "claude-3-5-sonnet": ("anthropic", "claude-3-5-sonnet-latest"),
    "claude-3-5-haiku": ("anthropic", "claude-3-5-haiku-latest"),
    "claude-3-opus": ("anthropic", "claude-3-opus-latest"),
    "llama-3.3-70b": ("groq", "llama-3.3-70b-versatile"),
    "llama-3.1-8b": ("groq", "llama-3.1-8b-instant"),
    "mixtral-8x7b": ("groq", "mixtral-8x7b-32768"),
}


def get_model(name: str, **kwargs) -> LLMProvider:
    """
    Get a pre-configured model by name.

    Args:
        name: Model shortcut name
        **kwargs: Additional configuration

    Returns:
        Configured LLMProvider
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")

    provider_type, model_id = MODELS[name]
    return create_provider(provider_type, model_id, **kwargs)
