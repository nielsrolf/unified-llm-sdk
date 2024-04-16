from any_llm.utils import merge_nested, stream


def test_conversation_with_text_messages():
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    model = "gpt-3.5-turbo"
    for message in stream(model, messages):
        pass
    assert len(message["content"]) > 0
    assert message["role"] == "assistant"
    messages.append(message)
    messages.append(
        {"role": "user", "content": "I just wanted to tell you you're awesome!"}
    )
    for message in stream(model, messages):
        pass
    assert len(message["content"]) > 0
    assert message["role"] == "assistant"


def test_merge_mested():
    message = {
        "content": None,
        "function_call": None,
        "role": "assistant",
        "tool_calls": [
            {
                "index": 0,
                "id": "call_VN0HNTbkXK4FXIx3dp9h88kU",
                "function": {"arguments": '{"', "name": "get_current_weather"},
                "type": "function",
            }
        ],
    }
    chunk = {
        "content": None,
        "function_call": None,
        "role": None,
        "tool_calls": [
            {
                "index": 0,
                "id": None,
                "function": {"arguments": "location", "name": None},
                "type": None,
            }
        ],
    }
    merged = merge_nested(message, chunk)
    assert merged == {
        "content": None,
        "function_call": None,
        "role": "assistant",
        "tool_calls": [
            {
                "index": 0,
                "id": "call_VN0HNTbkXK4FXIx3dp9h88kU",
                "function": {"arguments": '{"location', "name": "get_current_weather"},
                "type": "function",
            }
        ],
    }


def test_conversation_with_tool_calls():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
    model = "gpt-3.5-turbo"
    for message in stream(model, messages, tools=tools):
        pass
    assert len(message["tool_calls"]) > 0
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_current_weather"
    assert tool_call["function"]["arguments"]["location"] == "Boston"
    assert tool_call["function"]["arguments"]["unit"] in ["celsius", "fahrenheit"]
