# Any LLM

This repo contains a simple wrappers around openai's, gemini's and anthropic's sdks that allow easily switching or comparing all their models.

Features:
- prompt caching for cheaper and faster experimentation (just pass a seed)
- minimal agents
- function calling for Claude

Soon:
- streaming
- image inputs
- function calling for non-openai models

## Installation:
```
pip install git+https://github.com/nielsrolf/unified-llm-sdk
```

## Usage
```python
from any_llm import def get_response, get_agents

response = get_response(
    model="claude-3-opus-20240229",
    history=[{
        'role': 'system',
        'content': 'You are a helpful assistant'
    }],
    temperature=0,
    seed=2
)


agents = get_agents(models=anthr_models[1:] + openai_models, temperatures=[0])

```