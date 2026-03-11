# vLLM Client

A simple Python script to list all models and run a chat completion on a local [vLLM](https://github.com/vllm-project/vllm) server.

## Requirements

- Python 3.10+
- A running vLLM server at `http://localhost:8000`

## Install dependencies

```bash
uv pip install .
```

## Run

```bash
python main.py
```

To target a specific model:

```bash
python main.py "mistralai/Mistral-7B-Instruct-v0.2"
```

## Configuration

Edit the constants at the top of `main.py` to match your setup:

```python
export VLLM_BASE_URL="http://my-server:8000/v1"
export VLLM_API_KEY="my-secret-key"
python main.py

```