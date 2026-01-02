# AirOS (AI Reliability OS)

**A local-first, agentic reliability framework for LangGraph developers.**

AirOS acts as an "Active Reliability" layer that intercepts node execution to prevent loops, fix errors, and validate outputs. It is designed to be a zero-config, "drop-in" medic for your AI agents.

## 🚀 Features

- **The Fuse**: Automatically detects execution loops (e.g., repeating the same state 3 times) and trips to prevent infinite costs.
- **The Medic**: Catches exceptions (like JSON parsing errors) and attempts to repair them using intelligent strategies.
- **The Sentinel**: Validates node outputs against strict Pydantic schemas, ensuring downstream compatibility.
- **Local Dashboard**: A real-time "Pulse" view of your agent's execution, running locally on your machine.
- **Local-First Storage**: All traces are stored locally in `.air_os/traces.db` (SQLite). No data leaves your machine.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/air-os.git
cd air-os

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install .
```

## 🛠️ Usage

### 1. The SDK
Decorate your LangGraph nodes with `@reliable_node` to instantly add reliability features.

```python
from airos import reliable_node
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str

@reliable_node(fuse_limit=3, sentinel_schema=OutputSchema)
def my_agent_node(state):
    # Your compiled LangGraph logic here
    return {"summary": "This is reliable."}
```

### 2. The Dashboard
Launch the reliability control center to watch your agents think in real-time.

```bash
# Run from your project root
air-os dashboard
```
> Opens the dashboard at `http://localhost:3000`

## 🧪 Development

### Project Structure
- `airos/`: Core Python SDK.
- `dashboard/`: Next.js 15 + Tailwind CSS frontend.
- `.air_os/`: Local SQLite database (auto-generated).

### Running Tests
```bash
# Verify the core SDK functionality
python verify_script.py
```
