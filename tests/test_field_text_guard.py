"""Guards against meta-text being typed into real fields (user-verified bug).

The d1 fallback option "None of the above; ..." once leaked its own label as
the typed value on a Chinese goal. These tests pin the fix: refusal labels
never become values, Chinese goals yield real candidates, and the agent stops
bounded instead of looping when no value can be produced.
"""

import time
from unittest.mock import Mock

import pytest

from fast_browser_use import agent as loop
from fast_browser_use import model
from fast_browser_use.llamacpp_backend import LlamaCppModel

# --- is_garbage_field_text -------------------------------------------------


def test_refusal_label_is_garbage():
    assert model.is_garbage_field_text(
        "None of the above; the goal supplies no value for this field."
    )
    assert model.is_garbage_field_text("none of the above")


@pytest.mark.parametrize(
    "text",
    [
        "I should search for the channel",
        "The goal does not provide a value",
        "Let me think about it",
        "无法确定要输入的内容",
        "目标未提供关键词",
        "",
        None,
    ],
)
def test_reasoning_and_refusals_are_garbage(text):
    assert model.is_garbage_field_text(text)


@pytest.mark.parametrize("text", ["CyanXLab", "A browser is a choice, not a conversation", "生日快乐"])
def test_real_values_are_not_garbage(text):
    assert not model.is_garbage_field_text(text)


# --- extract_field_fragments -----------------------------------------------


def test_chinese_goal_yields_keyword_value():
    fragments = model.extract_field_fragments("去bilibili搜索CyanXLab，以及结合网络上搜到的信息给我一个信息画像")
    assert fragments[0] == "CyanXLab"


def test_chinese_goal_yields_search_site_value():
    fragments = model.extract_field_fragments("打开bilibili搜索PowerOverwhelming")
    assert "PowerOverwhelming" in fragments
    assert "bilibili" in fragments


def test_quoted_fragments_come_first():
    fragments = model.extract_field_fragments('搜索"cheap flights"然后打开第一个结果')
    assert fragments[0] == "cheap flights"


def test_english_goal_still_extracts():
    fragments = model.extract_field_fragments("Search for quantum computing on Wikipedia")
    assert any("quantum computing" in f for f in fragments)


def test_fragments_are_deduplicated_and_capped():
    fragments = model.extract_field_fragments("搜索ABC然后查找abc再输入ABC")
    lowers = [f.lower() for f in fragments]
    assert len(lowers) == len(set(lowers))
    assert len(fragments) <= 8


# --- _scored_field_text fallback handling ----------------------------------


def scored_backend(monkeypatch, choice):
    backend = object.__new__(LlamaCppModel)
    backend.labels = list("ABCDEFGH")
    captured = {}

    def fake_systemone(_context, request, purpose=None):
        captured["criteria"] = request["criteria"]
        return {"choice": choice}

    monkeypatch.setattr(backend, "_systemone", fake_systemone)
    return backend, captured


def test_fallback_choice_never_becomes_typed_value(monkeypatch):
    backend, captured = scored_backend(monkeypatch, "D")  # D is the fallback code
    with pytest.raises(ValueError, match="nothing typed"):
        backend._scored_field_text({"goal": "去bilibili搜索CyanXLab"})
    fallback_labels = [v for v in captured["criteria"].values() if "None of the above" in v]
    assert len(fallback_labels) == 1  # offered to the model, but never returned


def test_scored_field_text_returns_selected_value(monkeypatch):
    backend, captured = scored_backend(monkeypatch, "A")
    value = backend._scored_field_text({"goal": "去bilibili搜索CyanXLab，以及给我一个信息画像"})
    assert value == "CyanXLab"
    assert "None of the above" not in value


def test_unknown_choice_is_rejected(monkeypatch):
    backend, _ = scored_backend(monkeypatch, "Z")
    with pytest.raises(ValueError, match="nothing typed"):
        backend._scored_field_text({"goal": "搜索CyanXLab"})


# --- field_text choke point -------------------------------------------------


def test_field_text_rejects_garbage_from_backend(monkeypatch):
    garbage = "None of the above; the goal supplies no value for this field."
    fake = Mock()
    fake.generate_text = Mock(return_value=(garbage, {"model": "x"}))
    monkeypatch.setattr(model, "get_model", lambda: fake)
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "g"})


def test_field_text_passes_real_values_through(monkeypatch):
    helper = {"model": "x", "latency_ms": 1}
    fake = Mock()
    fake.generate_text = Mock(return_value=("CyanXLab", helper))
    monkeypatch.setattr(model, "get_model", lambda: fake)
    value, got = model.field_text({"goal": "g"})
    assert value == "CyanXLab" and got is helper


# --- agent loop: bounded stop instead of garbage typing ---------------------


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    p = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
        ],
    }
    p["fingerprint"] = __import__("fast_browser_use.browser", fromlist=["fingerprint"]).fingerprint(p)
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p), prepare=Mock(return_value=p)),
        "page": p,
        "decision": None,
        "goal": "搜索CyanXLab",
        "plan": ["搜索CyanXLab"],
        "plan_index": 0,
        "planned": True,
        "plan_calls": [],
        "milestones": [],
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def _fill_decision():
    return {
        "choice": "e1",
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 0.1,
        "probabilities": {"e1": 0.1},
        "latency_ms": 10,
        "usage": {},
    }


def test_no_value_failure_never_touches_the_browser(runner, monkeypatch):
    monkeypatch.setattr(loop, "field_text", Mock(side_effect=ValueError("nothing typed")))
    monkeypatch.setattr(loop, "choose", lambda *a, **k: _fill_decision())
    runner.state["decision"] = _fill_decision()
    runner.command("tick")
    runner.state["browser"].act.assert_not_called()
    assert runner.state["history"][-1]["kind"] == "no_value"
    assert runner.state["status"] == "ready"


def test_repeated_no_value_failures_stop_blocked(runner, monkeypatch):
    monkeypatch.setattr(loop, "field_text", Mock(side_effect=ValueError("nothing typed")))
    monkeypatch.setattr(loop, "choose", lambda *a, **k: _fill_decision())
    for _ in range(6):
        if runner.state["status"] == "blocked":
            break
        runner.state["decision"] = _fill_decision()
        runner.command("tick")
    assert runner.state["status"] == "blocked"
    assert "无法从目标中提取" in (runner.state.get("summary") or "")
    runner.state["browser"].act.assert_not_called()


def test_no_value_entries_do_not_trigger_stuck_rescue(runner, monkeypatch):
    monkeypatch.setattr(loop, "field_text", Mock(side_effect=ValueError("nothing typed")))
    monkeypatch.setattr(loop, "choose", lambda *a, **k: _fill_decision())
    monkeypatch.setattr(loop.Agent, "_rescue", Mock(side_effect=AssertionError("rescue must not fire")))
    for _ in range(3):
        runner.state["decision"] = _fill_decision()
        runner.command("tick")
    assert runner.state["status"] == "ready"
