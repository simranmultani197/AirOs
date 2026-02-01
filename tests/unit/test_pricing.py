"""
Unit tests for the Pricing module - Cost calculation and model pricing.
"""
import pytest

from airos.pricing import (
    ModelPricing,
    CostCalculator,
    MODEL_PRICING,
    DEFAULT_PRICING,
    get_model_pricing,
    estimate_tokens,
)


# ============================================================================
# ModelPricing Tests
# ============================================================================

class TestModelPricing:
    """Test ModelPricing dataclass."""

    def test_pricing_values(self):
        """Test pricing stores input and output rates."""
        p = ModelPricing(input_per_token=0.000003, output_per_token=0.000015)
        assert p.input_per_token == 0.000003
        assert p.output_per_token == 0.000015

    def test_avg_per_token(self):
        """Test average per token calculation."""
        p = ModelPricing(input_per_token=0.000002, output_per_token=0.000010)
        assert p.avg_per_token == 0.000006


# ============================================================================
# MODEL_PRICING Table Tests
# ============================================================================

class TestModelPricingTable:
    """Test the built-in pricing table."""

    def test_gpt4o_exists(self):
        """Test GPT-4o is in pricing table."""
        assert "gpt-4o" in MODEL_PRICING
        p = MODEL_PRICING["gpt-4o"]
        assert p.input_per_token > 0
        assert p.output_per_token > p.input_per_token

    def test_claude_exists(self):
        """Test Claude models are in pricing table."""
        assert "claude-3-5-sonnet" in MODEL_PRICING
        assert "claude-3-5-haiku" in MODEL_PRICING
        assert "claude-3-opus" in MODEL_PRICING

    def test_groq_exists(self):
        """Test Groq models are in pricing table."""
        assert "llama-3.3-70b-versatile" in MODEL_PRICING
        assert "llama-3.1-8b-instant" in MODEL_PRICING

    def test_pricing_table_not_empty(self):
        """Test pricing table has entries."""
        assert len(MODEL_PRICING) > 10

    def test_all_prices_positive(self):
        """Test all prices are positive."""
        for name, pricing in MODEL_PRICING.items():
            assert pricing.input_per_token > 0, f"{name} input price <= 0"
            assert pricing.output_per_token > 0, f"{name} output price <= 0"

    def test_output_more_expensive_for_major_models(self):
        """Test output is more expensive than input for major models."""
        for name in ["gpt-4o", "claude-3-5-sonnet", "claude-3-opus"]:
            p = MODEL_PRICING[name]
            assert p.output_per_token >= p.input_per_token, (
                f"{name}: output should cost >= input"
            )


# ============================================================================
# get_model_pricing Tests
# ============================================================================

class TestGetModelPricing:
    """Test model pricing lookup."""

    def test_exact_match(self):
        """Test exact model name match."""
        p = get_model_pricing("gpt-4o")
        assert p == MODEL_PRICING["gpt-4o"]

    def test_partial_match(self):
        """Test partial model name match."""
        p = get_model_pricing("gpt-4o-2024-08-06")
        # Should match gpt-4o pricing
        assert p.input_per_token == MODEL_PRICING["gpt-4o"].input_per_token

    def test_unknown_model_returns_default(self):
        """Test unknown model returns default pricing."""
        p = get_model_pricing("totally-unknown-model-xyz")
        assert p == DEFAULT_PRICING

    def test_case_insensitive_partial(self):
        """Test case-insensitive partial matching."""
        p = get_model_pricing("GPT-4O")
        # lowercase "gpt-4o" should be found in "gpt-4o" (the key)
        assert p.input_per_token > 0


# ============================================================================
# estimate_tokens Tests
# ============================================================================

class TestEstimateTokens:
    """Test token estimation."""

    def test_short_string(self):
        """Test token estimate for short string."""
        tokens = estimate_tokens("hello")
        assert tokens >= 1

    def test_long_string(self):
        """Test token estimate for long string."""
        tokens = estimate_tokens("x" * 1000)
        assert tokens == 250  # 1000 / 4

    def test_dict_input(self):
        """Test token estimate for dict."""
        tokens = estimate_tokens({"key": "value", "num": 42})
        assert tokens > 0

    def test_empty_string(self):
        """Test empty string returns at least 1."""
        tokens = estimate_tokens("")
        assert tokens >= 1

    def test_none_input(self):
        """Test None input."""
        tokens = estimate_tokens(None)
        assert tokens >= 1


# ============================================================================
# CostCalculator Tests
# ============================================================================

class TestCostCalculatorInit:
    """Test CostCalculator initialization."""

    def test_default_init(self):
        """Test default initialization uses default pricing."""
        calc = CostCalculator()
        assert calc.pricing == DEFAULT_PRICING

    def test_model_init(self):
        """Test initialization with model name."""
        calc = CostCalculator(model="gpt-4o")
        assert calc.pricing == MODEL_PRICING["gpt-4o"]

    def test_cost_per_token_init(self):
        """Test initialization with user override."""
        calc = CostCalculator(cost_per_token=0.00001)
        assert calc.pricing.input_per_token == 0.00001
        assert calc.pricing.output_per_token == 0.00001

    def test_cost_per_token_overrides_model(self):
        """Test cost_per_token takes priority over model."""
        calc = CostCalculator(model="gpt-4o", cost_per_token=0.00001)
        # Should use the override, not GPT-4o pricing
        assert calc.pricing.input_per_token == 0.00001

    def test_unknown_model_uses_default(self):
        """Test unknown model falls back to default."""
        calc = CostCalculator(model="unknown-model")
        assert calc.pricing == DEFAULT_PRICING


class TestCostCalculatorCalculate:
    """Test CostCalculator cost calculation."""

    def test_zero_tokens(self):
        """Test zero tokens costs zero."""
        calc = CostCalculator(model="gpt-4o")
        assert calc.calculate(0, 0) == 0.0

    def test_input_only(self):
        """Test cost with only input tokens."""
        calc = CostCalculator(model="gpt-4o")
        cost = calc.calculate(input_tokens=1000, output_tokens=0)
        expected = 1000 * MODEL_PRICING["gpt-4o"].input_per_token
        assert abs(cost - expected) < 1e-10

    def test_output_only(self):
        """Test cost with only output tokens."""
        calc = CostCalculator(model="gpt-4o")
        cost = calc.calculate(input_tokens=0, output_tokens=1000)
        expected = 1000 * MODEL_PRICING["gpt-4o"].output_per_token
        assert abs(cost - expected) < 1e-10

    def test_mixed_tokens(self):
        """Test cost with both input and output tokens."""
        calc = CostCalculator(model="gpt-4o")
        cost = calc.calculate(input_tokens=500, output_tokens=200)
        p = MODEL_PRICING["gpt-4o"]
        expected = 500 * p.input_per_token + 200 * p.output_per_token
        assert abs(cost - expected) < 1e-10

    def test_flat_rate_override(self):
        """Test flat rate applies equally to input and output."""
        calc = CostCalculator(cost_per_token=0.00001)
        cost = calc.calculate(input_tokens=500, output_tokens=500)
        assert abs(cost - 1000 * 0.00001) < 1e-10

    def test_gpt4o_vs_gpt4o_mini_cost_difference(self):
        """Test GPT-4o costs more than GPT-4o-mini."""
        calc_4o = CostCalculator(model="gpt-4o")
        calc_mini = CostCalculator(model="gpt-4o-mini")
        cost_4o = calc_4o.calculate(1000, 1000)
        cost_mini = calc_mini.calculate(1000, 1000)
        assert cost_4o > cost_mini

    def test_claude_opus_most_expensive(self):
        """Test Claude Opus is more expensive than Sonnet."""
        calc_opus = CostCalculator(model="claude-3-opus")
        calc_sonnet = CostCalculator(model="claude-3-5-sonnet")
        cost_opus = calc_opus.calculate(1000, 1000)
        cost_sonnet = calc_sonnet.calculate(1000, 1000)
        assert cost_opus > cost_sonnet


class TestCostCalculatorEstimateFromObjects:
    """Test CostCalculator.estimate_from_objects."""

    def test_simple_objects(self):
        """Test estimation from simple objects."""
        calc = CostCalculator(model="gpt-4o")
        tokens, cost = calc.estimate_from_objects(
            {"input": "test"},
            {"output": "result"}
        )
        assert tokens > 0
        assert cost > 0

    def test_large_objects_more_expensive(self):
        """Test larger objects cost more."""
        calc = CostCalculator(model="gpt-4o")
        _, cost_small = calc.estimate_from_objects("small", "tiny")
        _, cost_large = calc.estimate_from_objects("x" * 10000, "y" * 10000)
        assert cost_large > cost_small

    def test_returns_tuple(self):
        """Test returns (tokens, cost) tuple."""
        calc = CostCalculator()
        result = calc.estimate_from_objects("input", "output")
        assert isinstance(result, tuple)
        assert len(result) == 2
        tokens, cost = result
        assert isinstance(tokens, int)
        assert isinstance(cost, float)


class TestCostCalculatorRepr:
    """Test CostCalculator string representation."""

    def test_repr_default(self):
        """Test repr for default calculator."""
        calc = CostCalculator()
        assert "default" in repr(calc)

    def test_repr_model(self):
        """Test repr for model-based calculator."""
        calc = CostCalculator(model="gpt-4o")
        assert "gpt-4o" in repr(calc)

    def test_repr_override(self):
        """Test repr for override calculator."""
        calc = CostCalculator(cost_per_token=0.00001)
        assert "cost_per_token" in repr(calc)
