import json
import random

from any_llm.utils import fix_json


# Sample JSON strings for testing
def test_fix_json_partial():
    json_input = '{"name": "John", "age": 30'
    expected = {"name": "John", "age": 30}
    assert fix_json(json_input) == expected


def test_fix_json_correct():
    json_input = '{"name": "John", "age": 30}'
    expected = {"name": "John", "age": 30}
    assert fix_json(json_input) == expected


def test_fix_json_nested():
    json_input = '{"data": [{"id": 1, "value": "A"}, {"id": 2, "value": "B'
    expected = {"data": [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}]}
    assert fix_json(json_input) == expected


def test_fix_json_empty():
    json_input = ""
    expected = None
    assert fix_json(json_input) == expected


def build_nested_json(depth=4):
    data = {}
    for key in [0, 1, 2]:
        data[f"some-key-{key}"] = key
    if depth > 0:
        data[f"sublist{depth}"] = [build_nested_json(depth - 1) for _ in range(2)]
    return data


def test_nested_json():
    data = build_nested_json()
    data_str = json.dumps(data)
    for _ in range(10):
        broken = data_str[: random.randint(0, len(data_str))]
        fix_json(broken)
