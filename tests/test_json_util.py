"""Tests for the JSON extractor. No Ollama needed."""

from __future__ import annotations

from herding_cats.helpers.json_util import extract_json_object


def test_plain_json() -> None:
    text = '{"a": 1, "b": [2, 3]}'
    obj = extract_json_object(text)
    assert obj == {"a": 1, "b": [2, 3]}


def test_fenced_json() -> None:
    text = 'Sure, here you go:\n```json\n{"a": 2}\n```\nHope that helps!'
    obj = extract_json_object(text)
    assert obj == {"a": 2}


def test_fenced_json_uppercase_tag() -> None:
    text = '```JSON\n{"x": "y"}\n```'
    obj = extract_json_object(text)
    assert obj == {"x": "y"}


def test_embedded_in_prose() -> None:
    text = 'Here is what I think: {"answer": "yes", "reason": "obvious"} and that is it.'
    obj = extract_json_object(text)
    assert obj == {"answer": "yes", "reason": "obvious"}


def test_nested_braces_in_string() -> None:
    text = '{"msg": "hello {world}", "ok": true}'
    obj = extract_json_object(text)
    assert obj == {"msg": "hello {world}", "ok": True}


def test_garbage_returns_none() -> None:
    assert extract_json_object("") is None
    assert extract_json_object("just some prose, no braces") is None
    assert extract_json_object("{unbalanced") is None


def test_picks_first_balanced_object() -> None:
    text = 'Chatter {"first": 1} more chatter {"second": 2}'
    obj = extract_json_object(text)
    assert obj == {"first": 1}
