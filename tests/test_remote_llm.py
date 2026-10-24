"""Unit tests for the remote-LLM escalation module (ModelScope API-Inference)."""

import urllib.error

import pytest

from fast_browser_use import remote_llm


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("FBU_LLM_"):
            monkeypatch.delenv(key)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        import json

        return json.dumps(self._payload).encode("utf-8")


def _fake_urlopen(payload):
    def handler(request, timeout):
        return _FakeResponse(payload)

    return handler


def test_mode_and_enabled_defaults(monkeypatch):
    assert remote_llm.mode() == "auto"
    assert remote_llm.enabled() is True  # default key ships with the project


def test_enabled_off_and_empty_key(monkeypatch):
    monkeypatch.setenv("FBU_LLM_ESCALATION", "off")
    assert remote_llm.enabled() is False
    monkeypatch.setenv("FBU_LLM_ESCALATION", "auto")
    monkeypatch.setenv("FBU_LLM_API_KEY", "")
    assert remote_llm.enabled() is False


def test_confidence_threshold_auto_vs_explicit(monkeypatch):
    assert remote_llm.confidence_threshold() == 0.55
    monkeypatch.setenv("FBU_LLM_CONFIDENCE", "0.83")
    assert remote_llm.confidence_threshold() == 0.83


def test_chat_parses_openai_payload(monkeypatch):
    monkeypatch.setattr(
        remote_llm.urllib.request,
        "urlopen",
        _fake_urlopen({"choices": [{"message": {"content": "B"}}], "usage": {"prompt_tokens": 5}}),
    )
    content, telemetry = remote_llm.chat([{"role": "user", "content": "hi"}])
    assert content == "B"
    assert telemetry["backend"] == "remote-llm"
    assert telemetry["usage"]["prompt_tokens"] == 5


def test_chat_wraps_network_errors(monkeypatch):
    def boom(request, timeout):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(remote_llm.urllib.request, "urlopen", boom)
    with pytest.raises(remote_llm.RemoteLLMError):
        remote_llm.chat([{"role": "user", "content": "hi"}])


def test_decide_returns_valid_code(monkeypatch):
    monkeypatch.setattr(
        remote_llm, "chat",
        lambda messages, **kw: ("B", {"model": "m", "backend": "remote-llm", "latency_ms": 1, "usage": {}}),
    )
    record = remote_llm.decide("prompt", {"A": "first", "B": "second"})
    assert record["choice"] == "B"
    assert record["trigger"] == "low_confidence"


def test_decide_ignores_unknown_code(monkeypatch):
    monkeypatch.setattr(
        remote_llm, "chat",
        lambda messages, **kw: ("Z: no idea", {"model": "m", "backend": "remote-llm", "latency_ms": 1, "usage": {}}),
    )
    record = remote_llm.decide("prompt", {"A": "first", "B": "second"})
    assert record["choice"] is None  # kept as a record, but no valid override
    assert record["answer"].startswith("Z")


def test_consult_parses_answer_and_options(monkeypatch):
    reply = "ANSWER: The page introduces IANA example domains.\nOPTIONS: - Click More information"
    monkeypatch.setattr(
        remote_llm, "chat",
        lambda messages, **kw: (reply, {"model": "m", "backend": "remote-llm", "latency_ms": 1, "usage": {}}),
    )
    page = {"title": "t", "url": "u", "text": "body"}
    record = remote_llm.consult("task", "subgoal", page, [])
    assert record["answer"].startswith("The page introduces")
    assert record["options"] == ["Click More information"]
    assert record["trigger"] == "ask_llm"


def test_consult_answer_only(monkeypatch):
    reply = "ANSWER: All done, the form was saved."
    monkeypatch.setattr(
        remote_llm, "chat",
        lambda messages, **kw: (reply, {"model": "m", "backend": "remote-llm", "latency_ms": 1, "usage": {}}),
    )
    record = remote_llm.consult("task", "subgoal", {"title": "t", "url": "u", "text": ""}, [])
    assert record["answer"] == "All done, the form was saved."
    assert record["options"] == []


def test_rescue_stuck_parses_suggestion(monkeypatch):
    reply = '{"diagnosis": "clicked a dead link", "suggested_goal": "Use the search box instead"}'
    monkeypatch.setattr(
        remote_llm, "chat",
        lambda messages, **kw: (reply, {"model": "m", "backend": "remote-llm", "latency_ms": 1, "usage": {}}),
    )
    page = {"title": "t", "url": "u", "text": "body"}
    record = remote_llm.rescue_stuck("goal", [], page)
    assert record["suggested_goal"].startswith("Use the search box")
    assert record["trigger"] == "stuck"


def test_rescue_stuck_user_question(monkeypatch):
    reply = '{"need_user": true, "question": "Which city should I search?"}'
    monkeypatch.setattr(
        remote_llm, "chat",
        lambda messages, **kw: (reply, {"model": "m", "backend": "remote-llm", "latency_ms": 1, "usage": {}}),
    )
    record = remote_llm.rescue_stuck("goal", [], {"title": "t", "url": "u", "text": ""})
    assert record["trigger"] == "ask_user"
    assert "Which city" in record["question"]


def test_maybe_escalate_overrides_low_confidence_choice(monkeypatch):
    from fast_browser_use import model

    monkeypatch.setattr(remote_llm, "enabled", lambda: True)
    monkeypatch.setattr(remote_llm, "confidence_threshold", lambda: 0.9)
    monkeypatch.setattr(
        remote_llm, "decide",
        lambda prompt, options: {"trigger": "low_confidence", "choice": "B", "answer": "B", "model": "m"},
    )
    candidates = [
        {"id": "A", "operation": "CLICK", "target": None, "description": "first"},
        {"id": "B", "operation": "CLICK", "target": "btn", "description": "second"},
    ]
    result = {"choice": "A", "operation": "CLICK", "target": None, "confidence": 0.4, "probabilities": {}}
    result = model._maybe_escalate(result, "prompt", candidates, ["A", "B"])
    assert result["choice"] == "B"
    assert result["target"] == "btn"
    assert result["escalation"]["local_choice"] == "A"


def test_maybe_escalate_keeps_local_choice_when_confident(monkeypatch):
    from fast_browser_use import model

    monkeypatch.setattr(remote_llm, "enabled", lambda: True)
    monkeypatch.setattr(remote_llm, "confidence_threshold", lambda: 0.9)
    result = {"choice": "A", "operation": "CLICK", "target": None, "confidence": 0.95, "probabilities": {}}
    candidates = [{"id": "A", "operation": "CLICK", "target": None, "description": "first"}]
    assert model._maybe_escalate(result, "p", candidates, ["A"])["choice"] == "A"
    assert "escalation" not in result


def test_maybe_escalate_disabled_is_noop(monkeypatch):
    from fast_browser_use import model

    monkeypatch.setattr(remote_llm, "enabled", lambda: False)
    result = {"choice": "A", "operation": "CLICK", "target": None, "confidence": 0.1, "probabilities": {}}
    candidates = [{"id": "A", "operation": "CLICK", "target": None, "description": "first"}]
    assert model._maybe_escalate(result, "p", candidates, ["A"])["choice"] == "A"
    assert "escalation" not in result


def test_format_session_history_renders_turns():
    turns = [
        {"task": "打开 baidu.com 搜索新闻", "answer": "今日头条是…", "url": "https://www.baidu.com/s?wd=新闻"},
        {"task": "总结上一个结果", "answer": "要点有三：…", "url": "https://www.baidu.com"},
    ]
    block = remote_llm.format_session_history(turns)
    assert "USER TASK: 打开 baidu.com 搜索新闻" in block
    assert "LLM ANSWER: 今日头条是…" in block
    assert "PAGE REACHED: https://www.baidu.com" in block


def test_format_session_history_drops_oldest_and_truncates():
    turns = [{"task": f"task {i}", "answer": "a" * 80, "url": f"https://x/{i}"} for i in range(10)]
    block = remote_llm.format_session_history(turns, limit=3)
    assert "task 9" in block and "task 0" not in block
    short = remote_llm.format_session_history(turns, limit=6, chars=300)
    assert len(short) <= 300


def test_consult_includes_session_history(monkeypatch):
    captured = {}

    def fake_chat(messages, **_kwargs):
        captured["messages"] = messages
        return "ANSWER: ok", {"model": "m", "backend": "remote-llm", "device": "api", "latency_ms": 1.0, "usage": {}}

    monkeypatch.setattr(remote_llm, "chat", fake_chat)
    page = {"title": "Example", "url": "https://example.com", "text": "hello"}
    turns = [{"task": "先打开 example.com", "answer": "已打开", "url": "https://example.com"}]
    remote_llm.consult("总结", "总结", page, [], session_context=turns)
    user_content = captured["messages"][1]["content"]
    assert "SESSION HISTORY" in user_content
    assert "先打开 example.com" in user_content


def test_consult_without_session_history_has_no_block(monkeypatch):
    captured = {}

    def fake_chat(messages, **_kwargs):
        captured["messages"] = messages
        return "ANSWER: ok", {"model": "m", "backend": "remote-llm", "device": "api", "latency_ms": 1.0, "usage": {}}

    monkeypatch.setattr(remote_llm, "chat", fake_chat)
    page = {"title": "Example", "url": "https://example.com", "text": "hello"}
    remote_llm.consult("总结", "总结", page, [])
    assert "SESSION HISTORY" not in captured["messages"][1]["content"]


def test_rescue_includes_session_history(monkeypatch):
    captured = {}

    def fake_chat(messages, **_kwargs):
        captured["messages"] = messages
        return '{"diagnosis": "d", "suggested_goal": "g"}', {
            "model": "m", "backend": "remote-llm", "device": "api", "latency_ms": 1.0, "usage": {},
        }

    monkeypatch.setattr(remote_llm, "chat", fake_chat)
    page = {"title": "Example", "url": "https://example.com", "text": "hello"}
    record = remote_llm.rescue_stuck("目标", [], page, session_context=[{"task": "之前的任务"}])
    assert record["suggested_goal"] == "g"
    assert "之前的任务" in captured["messages"][1]["content"]


def test_agent_ask_llm_streak_guard(monkeypatch):
    """Consecutive ASK_LLM calls without a browser action must terminate the run."""
    from fast_browser_use import Agent

    consults = {"n": 0}

    def fake_consult(self, decision):
        consults["n"] += 1
        return {"trigger": "ask_llm", "answer": f"answer {consults['n']}", "options": ["try something"]}

    class FakeBrowser:
        def __init__(self, *_a, **_k):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(Agent, "_consult", fake_consult)
    monkeypatch.setattr("fast_browser_use.agent.Browser", FakeBrowser)
    agent = Agent.__new__(Agent)
    import time as _time

    agent.pending_text = None
    agent.record_dir = None
    agent.screenshots = False
    agent.browser = FakeBrowser()
    agent.state = dict(
        goal="g", page={"url": "https://x", "fingerprint": "f", "actions": [], "title": "t", "text": ""},
        decision=None, history=[], status="ready", plan=["g"], plan_index=0, step_history_index=0,
        planned=True, plan_calls=[], milestones=[], decisions=[], rejections=[], escalations=[],
        answers=[], ask_llm_streak=0, session_history=[], observations=[], text_calls=[],
        elapsed_ms=0, started_at=_time.perf_counter(), record=False,
    )
    for _ in range(5):
        agent.state["decision"] = {
            "choice": "ASK_LLM", "probabilities": {"ASK_LLM": 0.9}, "confidence": 0.9,
            "latency_ms": 1, "usage": {}, "operation": "ASK_LLM", "target": None,
        }
        agent.command("act", {"fingerprint": "f"})
        if agent.state["status"] in {"done", "blocked"}:
            break
    assert agent.state["status"] in {"done", "blocked"}
    assert consults["n"] <= 3  # limit=2 → the 3rd consult ends the run
