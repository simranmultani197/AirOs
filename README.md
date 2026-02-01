# AirOS

**One decorator to make any AI agent reliable.**

```
pip install airos-sdk
```

AirOS wraps your AI agent functions with invisible safety nets:

- **Fuse** - Detects infinite loops and kills them before they drain your wallet
- **Medic** - Catches exceptions and auto-repairs outputs using an LLM
- **Sentinel** - Validates every output against your Pydantic schema
- **Budget** - Dollar and time circuit breakers that stop runaway costs
- **Pricing** - Accurate cost tracking with built-in pricing for 25+ models

Zero config. No server. No database. Just a decorator.

---

## Before / After

**Before** - your agent crashes, loops forever, or returns garbage:

```python
def extract_data(state):
    result = call_llm(state["text"])
    return json.loads(result)  # crashes on malformed JSON
```

**After** - one line change, your agent self-heals:

```python
from airos import reliable
from pydantic import BaseModel

class Output(BaseModel):
    name: str
    age: int

@reliable(sentinel_schema=Output)
def extract_data(state):
    result = call_llm(state["text"])
    return json.loads(result)  # if this crashes, Medic fixes it
```

What happens behind the scenes:
1. **Budget** checks if you've exceeded your dollar or time limit
2. **Fuse** checks if this node is stuck in a loop (same input seen 3+ times)
3. Your function runs normally
4. **Sentinel** validates the output against `Output` schema
5. If anything fails, **Medic** calls an LLM to fix the output
6. Cost is tracked using **Pricing** (model-aware or user-configured)
7. If Medic fails twice, the error propagates (no silent failures)

---

## Quick Start

### Minimal (no LLM, just validation + loop detection)

```python
from airos import reliable
from pydantic import BaseModel

class SearchResult(BaseModel):
    query: str
    results: list[str]

@reliable(sentinel_schema=SearchResult, fuse_limit=5)
def search_node(state):
    return {"query": state["q"], "results": ["result1", "result2"]}
```

### With Auto-Repair (add an LLM for self-healing)

```
pip install airos-sdk[groq]  # or airos-sdk[openai] or airos-sdk[anthropic]
```

```python
import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def my_llm(prompt: str) -> str:
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    ).choices[0].message.content

@reliable(sentinel_schema=SearchResult, llm_callable=my_llm)
def search_node(state):
    return {"query": state["q"], "results": ["result1", "result2"]}
```

Now if `search_node` throws an exception or returns invalid data, Medic will use your LLM to generate a valid output matching the schema.

---

## Cost Saving

AirOS provides three layers of cost protection to prevent runaway agent loops from draining your wallet.

### Per-Node Dollar Limit

Stop a single node from spending too much:

```python
from airos import reliable

@reliable(max_cost_usd=2.0, model="gpt-4o")
def expensive_node(state):
    return call_llm(state["prompt"])
# Raises BudgetExceededError if this node's cumulative cost exceeds $2.00
```

### Per-Node Time Limit

Kill long-running nodes before they hang your pipeline:

```python
@reliable(max_seconds=30)
def slow_node(state):
    return call_external_api(state["query"])
# Raises TimeoutExceededError if execution exceeds 30 seconds
```

### Global Budget (Shared Across Nodes)

Set one budget for your entire agent graph — all nodes share it:

```python
from airos import reliable, GlobalBudget

budget = GlobalBudget(max_cost_usd=10.0, max_seconds=120)

@reliable(budget=budget, model="gpt-4o")
def node_a(state):
    return call_llm_a(state)

@reliable(budget=budget, model="gpt-4o-mini")
def node_b(state):
    return call_llm_b(state)

# Run your pipeline...
result_a = node_a({"prompt": "analyze this"})
result_b = node_b({"prompt": "summarize that"})

# Check spend after execution
print(f"Total spent: ${budget.total_spent:.4f}")
print(f"Remaining:   ${budget.remaining:.4f}")
print(f"Elapsed:     {budget.elapsed_seconds:.1f}s")

# Reset for next run
budget.reset()
```

If the combined cost of `node_a` + `node_b` exceeds $10.00 or 120 seconds, the next node call raises an error immediately — before making another LLM call.

### Catching Budget Errors

```python
from airos import reliable, GlobalBudget, BudgetExceededError, TimeoutExceededError

budget = GlobalBudget(max_cost_usd=5.0, max_seconds=60)

@reliable(budget=budget, model="gpt-4o")
def my_node(state):
    return call_llm(state)

try:
    for task in tasks:
        my_node(task)
except BudgetExceededError as e:
    print(f"Budget hit: spent ${e.spent:.2f} of ${e.limit:.2f} limit")
except TimeoutExceededError as e:
    print(f"Timeout hit: {e.elapsed:.1f}s of {e.limit:.1f}s limit")
```

---

## Pricing & Cost Tracking

AirOS automatically tracks the cost of every node execution. It uses a priority chain to get the most accurate cost:

| Priority | Source | Accuracy |
|----------|--------|----------|
| 1 | Actual API token usage (from AirOS providers) | Exact |
| 2 | User-provided `cost_per_token` | Explicit |
| 3 | Built-in model pricing table lookup | Good estimate |
| 4 | Default $5/1M tokens flat rate | Rough fallback |

### Specify Your Model (recommended)

```python
@reliable(model="gpt-4o")
def my_node(state):
    return call_openai(state)
# Cost is calculated using GPT-4o's actual pricing:
# $2.50/1M input tokens, $10.00/1M output tokens
```

### Custom Cost Per Token

Override with your own rate (e.g., for fine-tuned models or enterprise pricing):

```python
@reliable(cost_per_token=0.00003)  # $30/1M tokens
def my_node(state):
    return call_custom_model(state)
```

### Supported Models

Built-in pricing for 25+ models across major providers:

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o3-mini |
| **Anthropic** | claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, claude-3-haiku, claude-3-sonnet |
| **Groq** | llama-3.3-70b, llama-3.1-8b, mixtral-8x7b, llama-3.1-70b, gemma2-9b |
| **Google** | gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash |

Model names are matched flexibly — `"gpt-4o"`, `"gpt-4o-2024-08-06"`, and `"GPT-4o"` all resolve to the same pricing.

### Using CostCalculator Directly

```python
from airos import CostCalculator, get_model_pricing

# Check a model's pricing
pricing = get_model_pricing("gpt-4o")
print(f"Input:  ${pricing.input_per_token * 1_000_000:.2f}/1M tokens")
print(f"Output: ${pricing.output_per_token * 1_000_000:.2f}/1M tokens")

# Calculate cost for known token counts
calc = CostCalculator(model="gpt-4o")
cost = calc.calculate(input_tokens=1000, output_tokens=500)
print(f"Cost: ${cost:.6f}")
```

---

## With LangGraph

```python
from langgraph.graph import StateGraph
from airos import reliable, GlobalBudget
from pydantic import BaseModel

class AgentState(BaseModel):
    messages: list[str]
    result: str = ""

budget = GlobalBudget(max_cost_usd=5.0, max_seconds=120)

@reliable(sentinel_schema=AgentState, fuse_limit=3, budget=budget, model="gpt-4o")
def process_node(state):
    # Your agent logic
    return {"messages": state["messages"], "result": "done"}

graph = StateGraph(AgentState)
graph.add_node("process", process_node)
```

## With LangChain / CrewAI / AutoGen

```python
from airos import get_adapter

# LangChain
adapter = get_adapter("langchain", fuse_limit=5)
wrapped_chain = adapter.wrap_chain(my_chain, schema=OutputSchema)

# CrewAI
adapter = get_adapter("crewai")
wrapped_agent = adapter.wrap_agent(my_agent)

# AutoGen
adapter = get_adapter("autogen")
wrapped_function = adapter.wrap_function(my_tool)
```

---

## How It Works

```
Your Function
     |
     v
  [ Budget ] -- Over budget? --> STOP (BudgetExceededError)
     |
     v
  [ Fuse ] -- Loop detected? --> STOP (LoopError)
     |
     v
  Run your function
     |
     v
  [ Pricing ] -- Track cost (model-aware)
     |
     v
  [ Sentinel ] -- Output valid? --> Return result
     |
     v (invalid)
  [ Medic ] -- LLM fixes output --> [ Sentinel ] validates again
     |
     v (still fails after 2 attempts)
  Raise original error (no silent failures)
```

## Storage

By default, AirOS stores traces **in memory** (lost when process exits). For persistence:

```python
from airos import reliable, set_default_storage
from airos.storage import Storage  # SQLite backend

set_default_storage(Storage())  # Now traces persist to .air_os/traces.db

@reliable()
def my_node(state):
    return state
```

Or pass storage per-decorator:

```python
from airos.storage import Storage

db = Storage(db_path="my_traces.db")

@reliable(storage=db)
def my_node(state):
    return state
```

## Installation Options

```bash
pip install airos-sdk                 # Core only (pydantic). Validation + loop detection.
pip install airos-sdk[groq]           # + Groq for auto-repair
pip install airos-sdk[openai]         # + OpenAI for auto-repair
pip install airos-sdk[anthropic]      # + Anthropic for auto-repair
pip install airos-sdk[langchain]      # + LangChain adapter
pip install airos-sdk[crewai]         # + CrewAI adapter
pip install airos-sdk[llm]            # All LLM providers
pip install airos-sdk[all]            # Everything
```

## API Reference

### `@reliable()` / `@reliable_node()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentinel_schema` | `BaseModel` | `None` | Pydantic model for output validation |
| `llm_callable` | `Callable[[str], str]` | `None` | LLM function for auto-repair |
| `fuse_limit` | `int` | `3` | Max identical states before loop detection |
| `node_name` | `str` | `None` | Override function name for traces |
| `storage` | `BaseStorage` | `InMemoryStorage` | Storage backend for traces |
| `medic_repair` | `Callable` | `None` | Custom repair callback (legacy) |
| `max_cost_usd` | `float` | `None` | Per-node dollar budget limit |
| `max_seconds` | `float` | `None` | Per-node execution time limit (seconds) |
| `budget` | `GlobalBudget` | `None` | Shared budget across multiple nodes |
| `model` | `str` | `None` | Model name for pricing table lookup |
| `cost_per_token` | `float` | `None` | Custom cost per token override (USD) |

### Core Components

```python
from airos import Fuse, Medic, Sentinel  # Use individually if needed
```

- **`Fuse(limit=3)`** - Loop detection via state hashing
- **`Medic(llm_callable=...)`** - LLM-based error recovery
- **`Sentinel(schema=...)`** - Pydantic schema validation

### Budget Components

```python
from airos import BudgetFuse, TimeoutFuse, GlobalBudget
```

- **`BudgetFuse(max_cost_usd=1.0)`** - Dollar-based circuit breaker
- **`TimeoutFuse(max_seconds=30)`** - Time-based circuit breaker
- **`GlobalBudget(max_cost_usd=10.0, max_seconds=120)`** - Thread-safe shared budget

### Pricing Components

```python
from airos import CostCalculator, get_model_pricing, MODEL_PRICING
```

- **`CostCalculator(model="gpt-4o")`** - Calculate costs using model pricing
- **`CostCalculator(cost_per_token=0.00003)`** - Calculate costs using custom rate
- **`get_model_pricing("gpt-4o")`** - Look up a model's pricing
- **`MODEL_PRICING`** - Full pricing table dict

### Error Types

```python
from airos import BudgetExceededError, TimeoutExceededError, LoopError
```

| Error | Raised When | Attributes |
|-------|-------------|------------|
| `BudgetExceededError` | Dollar limit exceeded | `spent`, `limit` |
| `TimeoutExceededError` | Time limit exceeded | `elapsed`, `limit` |
| `LoopError` | Infinite loop detected | — |

## Running Tests

```bash
pip install airos-sdk[dev]
pytest
```

## License

MIT
