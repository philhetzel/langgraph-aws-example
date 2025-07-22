# LangGraph AWS Bedrock Guardrails Threading Issue

This repository demonstrates and provides solutions for the threading issue that occurs when using AWS Bedrock Guardrails with LangGraph.

## The Problem

When using LangGraph with AWS Bedrock and Guardrails, you may encounter this error:

```
ValueError: signal only works in main thread of the main interpreter
During task with name 'guardrails_in' and id '6f38c2e0-97e6-98a4-8c57-acc744dc09f7'
```

This happens because:
1. LangGraph executes nodes in separate threads for async operations
2. AWS Bedrock Guardrails (via boto3) attempts to register signal handlers
3. Signal handlers can only be registered from the main thread

## Files

- `main.py` - Reproduces the original threading issue
- `fixed_main.py` - Contains multiple approaches to fix the issue
- `.env.example` - Environment variable template

## Setup

1. Install dependencies:
```bash
pip install boto3 braintrust langchain-aws langchain-core langgraph
```

2. Copy `.env.example` to `.env` and fill in your AWS credentials:
```bash
cp .env.example .env
```

3. Set up your Bedrock Guardrail in AWS Console and get the guardrail ID

## Solutions

### Solution 1: Thread-Safe Bedrock Wrapper
Use `ThreadSafeBedrock` class in `fixed_main.py` that:
- Detects when running in non-main thread
- Uses `asyncio.run_coroutine_threadsafe()` to execute in main thread
- Handles initialization with thread-safe locking

### Solution 2: Remove Guardrails from Model Config
If threading issues persist:
1. Remove guardrailConfig from ChatBedrock initialization
2. Implement content filtering at application level
3. Use separate validation service calls

### Solution 3: LangGraph Configuration
Configure LangGraph to use main thread for specific nodes:
```python
# Use synchronous execution for problematic nodes
app = workflow.compile(checkpointer=None, interrupt_before=["guardrails_in"])
```

## Running the Examples

### 1. Reproduce the Original Issue:
```bash
uv run python main.py
```

### 2. See Threading Error in Isolation:
```bash
uv run python force_threading_test.py
```

### 3. Complete Demo (Problem + All Solutions):
```bash
uv run python complete_demo.py
```

### 4. Test Fixed Implementation:
```bash
uv run python fixed_main.py
```

## Additional Debugging

For Braintrust.dev customers, this issue commonly occurs during:
- Model evaluation with LangGraph agents
- Concurrent execution of multiple test cases
- Integration with AWS Bedrock's content filtering

The fixes provided should resolve the threading conflicts while maintaining Bedrock Guardrails functionality.