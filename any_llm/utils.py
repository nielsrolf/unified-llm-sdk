import json

from openai import OpenAI
from retry import retry


def fix_json(json_string):
    """parse partially streamed json"""
    if json_string is None:
        return None
    stack = []
    inside_string = False
    escape = False

    for char in json_string:
        if char == '"' and not escape:
            inside_string = not inside_string
            if inside_string:
                stack.append('"')
            else:
                stack.pop()

        if not inside_string:
            if char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif (char == "}" or char == "]") and stack:
                stack.pop()

        escape = char == "\\" and not escape

    # Append all unclosed elements in reverse order

    closing = "".join(stack[::-1])
    while len(json_string) > 0:
        try:
            val = json.loads(json_string + closing)
            return val
        except json.JSONDecodeError:
            json_string = json_string[:-1]
        except TypeError:
            breakpoint()
            print(json_string)


def merge_nested(message, chunk):
    """Merge nested jsons"""
    if chunk is None:
        return message
    if not message:
        return chunk
    if isinstance(message, dict):
        for key, value in chunk.items():
            if key in message:
                message[key] = merge_nested(message[key], value)
            else:
                message[key] = value
        return message
    elif isinstance(message, list):
        for i, value in enumerate(chunk):
            if i < len(message):
                message[i] = merge_nested(message[i], value)
            else:
                message.append(value)
        return message
    elif isinstance(message, str):
        if isinstance(chunk, str):
            return message + chunk
    breakpoint()
    return chunk


def fix_json_nested(data, key_regex):
    """Calls fix_json on specified keys"""
    if len(key_regex) == 0:
        return fix_json(data)
    key, remaining = key_regex[0], key_regex[1:]
    if key == "*":
        if isinstance(data, dict):
            data = data.copy()
            for k, v in data.items():
                data[k] = fix_json_nested(v, remaining)
        elif isinstance(data, list):
            return [fix_json_nested(v, remaining) for v in data]
    else:
        data = data.copy()
        data[key] = fix_json_nested(data.get(key, ""), remaining)
    return data


def remove_none_keys(message):
    return {k: v for k, v in message.items() if v is not None}


@retry(tries=3, delay=1)
def stream(
    model,
    messages,
    when_done=None,
    fix_json_keys=["tool_calls/*/function/arguments"],
    **completions_kwargs
):
    """Stream that parsed partial json after every new chunk and calls json_clb"""
    oai = OpenAI()
    messages = [remove_none_keys(msg) for msg in messages]
    response_stream = oai.chat.completions.create(
        model=model, messages=messages, stream=True, **completions_kwargs
    )
    message = {}
    for chunk in response_stream:
        message = merge_nested(message, chunk.choices[0].delta.model_dump())
        message_parsed = message.copy()

        for key in fix_json_keys:
            message_parsed = fix_json_nested(message_parsed, key.split("/"))
        yield message_parsed

    if when_done:
        when_done(message_parsed)
