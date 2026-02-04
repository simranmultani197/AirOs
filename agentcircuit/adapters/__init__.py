"""
Framework adapters for AgentCircuit.

Provides integration with multiple AI agent frameworks:
- LangGraph (native)
- LangChain
- CrewAI
- AutoGen
"""

from .base import FrameworkAdapter, AdapterConfig, AdapterRegistry
from .langgraph import LangGraphAdapter
from .langchain import LangChainAdapter
from .crewai import CrewAIAdapter
from .autogen import AutoGenAdapter

__all__ = [
    "FrameworkAdapter",
    "AdapterConfig",
    "AdapterRegistry",
    "LangGraphAdapter",
    "LangChainAdapter",
    "CrewAIAdapter",
    "AutoGenAdapter",
]
