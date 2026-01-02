from airos import reliable_node, LoopError, SentinelError
from pydantic import BaseModel
import json

# Sentinel Schema
class OutputModel(BaseModel):
    message: str
    value: int

# Mock LLM for Repair
def mock_repair_llm(prompt: str) -> str:
    print(f"\n[MockLLM] Received Prompt:\n{prompt[:100]}...\n")
    # Simulate fixing the error by returning valid JSON
    return json.dumps({"message": "Repaired by LLM", "value": 100})

# 1. Test Normal Execution
@reliable_node(node_name="test_node_normal")
def normal_node(state):
    print("Executing normal node...")
    return {"message": "Success", "value": 42}

# 2. Test Error Recovery (Medic with LLM)
@reliable_node(llm_callable=mock_repair_llm, sentinel_schema=OutputModel, node_name="test_node_medic_llm")
def error_node_llm(state):
    print("Executing error node (will fail)...")
    raise ValueError("Simulation: Invalid JSON or Logic Error")

# 3. Test Validation Recovery (Medic with LLM)
@reliable_node(llm_callable=mock_repair_llm, sentinel_schema=OutputModel, node_name="test_node_sentinel_llm")
def bad_schema_node(state):
    print("Executing bad schema node...")
    return {"message": "Fail", "value": "not an int"}

import time

def run_verification():
    # Use a unique suffix to avoid "Loop Detected" when running this script multiple times
    # because the database persists history.
    suffix = int(time.time())
    
    print("--- 1. Normal Run ---")
    run_id_1 = f"verify_mvp_{suffix}"
    print(f"Run ID: {run_id_1}")
    res = normal_node({"key": "val"}, config={"run_id": run_id_1})
    print(f"Result: {res}")

    print("\n--- 2. Medic Logic Repair (Exception) ---")
    run_id_2 = f"verify_medic_logic_{suffix}"
    print(f"Run ID: {run_id_2}")
    res = error_node_llm({"step": "error"}, config={"run_id": run_id_2})
    print(f"Result after medic: {res}")

    print("\n--- 3. Medic Schema Repair (Sentinel Error) ---")
    run_id_3 = f"verify_medic_schema_{suffix}"
    print(f"Run ID: {run_id_3}")
    res = bad_schema_node({"step": "bad_schema"}, config={"run_id": run_id_3})
    print(f"Result after medic: {res}")

if __name__ == "__main__":
    run_verification()
