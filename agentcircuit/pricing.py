"""
Model pricing table and cost calculation for AgentCircuit.

Provides accurate cost estimation based on:
1. Actual API token usage (when available from providers)
2. User-provided cost_per_token override
3. Built-in pricing table lookup by model name
4. Rough fallback estimate ($5/1M tokens)
"""
from dataclasses import dataclass
from typing import Any, Optional, Dict, Tuple


@dataclass
class ModelPricing:
    """Pricing for a specific model (USD per token)."""
    input_per_token: float
    output_per_token: float

    @property
    def avg_per_token(self) -> float:
        """Average of input and output cost (simple fallback)."""
        return (self.input_per_token + self.output_per_token) / 2


# Built-in pricing table (USD per token)
# Sources: official pricing pages as of Feb 2026
# These are approximate — providers update pricing regularly
MODEL_PRICING: Dict[str, ModelPricing] = {
    # OpenAI - Latest
    "gpt-5": ModelPricing(1.25 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4.5-preview": ModelPricing(75.00 / 1_000_000, 150.00 / 1_000_000),
    "o3": ModelPricing(2.00 / 1_000_000, 8.00 / 1_000_000),
    "o3-pro": ModelPricing(20.00 / 1_000_000, 80.00 / 1_000_000),
    "o3-mini": ModelPricing(1.10 / 1_000_000, 4.40 / 1_000_000),
    "o1": ModelPricing(15.00 / 1_000_000, 60.00 / 1_000_000),
    "o1-mini": ModelPricing(3.00 / 1_000_000, 12.00 / 1_000_000),
    # OpenAI - Previous Gen
    "gpt-4o": ModelPricing(2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-mini": ModelPricing(0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4-turbo": ModelPricing(10.00 / 1_000_000, 30.00 / 1_000_000),
    "gpt-4": ModelPricing(30.00 / 1_000_000, 60.00 / 1_000_000),
    "gpt-3.5-turbo": ModelPricing(0.50 / 1_000_000, 1.50 / 1_000_000),

    # Anthropic - Latest (Claude 4.x)
    "claude-opus-4.5": ModelPricing(5.00 / 1_000_000, 25.00 / 1_000_000),
    "claude-sonnet-4.5": ModelPricing(3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-haiku-4.5": ModelPricing(1.00 / 1_000_000, 5.00 / 1_000_000),
    "claude-opus-4.1": ModelPricing(15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-opus-4": ModelPricing(15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-sonnet-4": ModelPricing(3.00 / 1_000_000, 15.00 / 1_000_000),
    # Anthropic - Previous Gen (Claude 3.x)
    "claude-3-5-sonnet": ModelPricing(3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-3-5-sonnet-latest": ModelPricing(3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-3-5-haiku": ModelPricing(0.80 / 1_000_000, 4.00 / 1_000_000),
    "claude-3-5-haiku-latest": ModelPricing(0.80 / 1_000_000, 4.00 / 1_000_000),
    "claude-3-opus": ModelPricing(15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-3-opus-latest": ModelPricing(15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-3-haiku": ModelPricing(0.25 / 1_000_000, 1.25 / 1_000_000),
    "claude-3-sonnet": ModelPricing(3.00 / 1_000_000, 15.00 / 1_000_000),

    # Google - Latest (Gemini 3.x & 2.5)
    "gemini-3-pro": ModelPricing(2.00 / 1_000_000, 12.00 / 1_000_000),
    "gemini-3-flash": ModelPricing(0.50 / 1_000_000, 3.00 / 1_000_000),
    "gemini-2.5-pro": ModelPricing(1.25 / 1_000_000, 10.00 / 1_000_000),
    "gemini-2.5-flash": ModelPricing(0.30 / 1_000_000, 2.50 / 1_000_000),
    "gemini-2.5-flash-lite": ModelPricing(0.10 / 1_000_000, 0.40 / 1_000_000),
    # Google - Previous Gen
    "gemini-2.0-flash": ModelPricing(0.10 / 1_000_000, 0.40 / 1_000_000),
    "gemini-1.5-pro": ModelPricing(1.25 / 1_000_000, 5.00 / 1_000_000),
    "gemini-1.5-flash": ModelPricing(0.075 / 1_000_000, 0.30 / 1_000_000),

    # Groq (hosted models)
    "llama-3.3-70b": ModelPricing(0.59 / 1_000_000, 0.79 / 1_000_000),
    "llama-3.3-70b-versatile": ModelPricing(0.59 / 1_000_000, 0.79 / 1_000_000),
    "llama-3.1-8b": ModelPricing(0.05 / 1_000_000, 0.08 / 1_000_000),
    "llama-3.1-8b-instant": ModelPricing(0.05 / 1_000_000, 0.08 / 1_000_000),
    "mixtral-8x7b": ModelPricing(0.24 / 1_000_000, 0.24 / 1_000_000),
    "mixtral-8x7b-32768": ModelPricing(0.24 / 1_000_000, 0.24 / 1_000_000),
    "llama-3.1-70b-versatile": ModelPricing(0.59 / 1_000_000, 0.79 / 1_000_000),
    "gemma2-9b-it": ModelPricing(0.20 / 1_000_000, 0.20 / 1_000_000),
}

# Default fallback: $5 per 1M tokens (flat rate for input and output)
DEFAULT_PRICING = ModelPricing(5.00 / 1_000_000, 5.00 / 1_000_000)


def get_model_pricing(model_name: str) -> ModelPricing:
    """
    Look up pricing for a model by name.

    Tries exact match first, then partial match (e.g. "gpt-4o" matches "gpt-4o").

    Args:
        model_name: Model identifier string

    Returns:
        ModelPricing for the model, or DEFAULT_PRICING if not found
    """
    # Exact match
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Partial match (model name contains a known key)
    model_lower = model_name.lower()
    for key, pricing in MODEL_PRICING.items():
        if key in model_lower or model_lower in key:
            return pricing

    return DEFAULT_PRICING


def estimate_tokens(obj: Any) -> int:
    """
    Rough token estimation from any object.

    Uses the ~4 characters per token heuristic.
    This is a fallback when actual token counts aren't available.

    Args:
        obj: Any object to estimate token count for

    Returns:
        Estimated token count
    """
    s = str(obj)
    return max(1, len(s) // 4)


class CostCalculator:
    """
    Calculates execution costs using a priority-based approach:

    1. Actual API token usage (from LLMProvider.last_usage) — exact
    2. User-provided cost_per_token — explicit override
    3. Model pricing table lookup — good estimate
    4. Default $5/1M flat rate — rough fallback

    Usage:
        # Auto-detect from provider
        calc = CostCalculator(model="gpt-4o")

        # User override
        calc = CostCalculator(cost_per_token=0.00001)

        # Calculate
        cost = calc.calculate(input_tokens=500, output_tokens=200)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        cost_per_token: Optional[float] = None,
    ):
        """
        Args:
            model: Model name for pricing table lookup
            cost_per_token: User-provided flat rate override (USD per token)
        """
        self.model = model
        self.cost_per_token = cost_per_token
        self._pricing: Optional[ModelPricing] = None

        if cost_per_token is not None:
            # User override — use flat rate for both input and output
            self._pricing = ModelPricing(cost_per_token, cost_per_token)
        elif model:
            # Look up from pricing table
            self._pricing = get_model_pricing(model)
        else:
            # Default fallback
            self._pricing = DEFAULT_PRICING

    @property
    def pricing(self) -> ModelPricing:
        """Get the active pricing."""
        return self._pricing

    def calculate(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        """
        Calculate cost for given token counts.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens

        Returns:
            Estimated cost in USD
        """
        input_cost = input_tokens * self._pricing.input_per_token
        output_cost = output_tokens * self._pricing.output_per_token
        return input_cost + output_cost

    def estimate_from_objects(
        self,
        input_obj: Any,
        output_obj: Any,
    ) -> Tuple[int, float]:
        """
        Estimate tokens and cost from arbitrary objects.

        Args:
            input_obj: The input state/data
            output_obj: The output result/data

        Returns:
            Tuple of (estimated_tokens, estimated_cost)
        """
        input_tokens = estimate_tokens(input_obj)
        output_tokens = estimate_tokens(output_obj)
        total_tokens = input_tokens + output_tokens
        cost = self.calculate(input_tokens, output_tokens)
        return total_tokens, cost

    def __repr__(self) -> str:
        if self.cost_per_token is not None:
            return f"CostCalculator(cost_per_token={self.cost_per_token})"
        if self.model:
            return f"CostCalculator(model={self.model!r})"
        return "CostCalculator(default)"
