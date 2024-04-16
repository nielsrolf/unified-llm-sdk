from any_llm.utils import merge_nested


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
    breakpoint()
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
