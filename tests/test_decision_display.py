"""Decision-transcript display and llama-server error resilience (no network)."""

import json
import urllib.error
from unittest import mock

from fast_browser_use import chat, tui
from fast_browser_use import llamacpp_backend as lb


def _state():
    return {
        "page": {
            "actions": [
                {"id": "e1", "kind": "click", "label": "搜索"},
                {"id": "e2", "kind": "click", "label": "More information"},
                {"id": "scroll_down", "kind": "scroll", "label": "Scroll down"},
            ]
        }
    }


# ------------------------------------------------------------- display ----

def test_candidate_ranked_orders_and_labels():
    decision = {
        "probabilities": {"e2": 0.41, "e1": 0.12, "DONE": 0.08, "scroll_down": 0.05, "ASK_LLM": 0.03},
        "scoring_passes": 2,
    }
    ranked = chat._candidate_ranked(_state(), decision, "e2")
    assert ranked[0] == ("click「More information」", 0.41)
    assert ranked[1] == ("click「搜索」", 0.12)
    assert len(ranked) == 3


def test_chosen_candidate_outside_top_is_kept():
    decision = {"probabilities": {"e1": 0.5, "e2": 0.3, "DONE": 0.1, "scroll_down": 0.05, "ASK_LLM": 0.02}}
    ranked = chat._candidate_ranked(_state(), decision, "ASK_LLM")
    assert ranked[-1][0] == "ASK_LLM"
    assert len(ranked) == 4


def test_long_labels_are_truncated():
    state = {"page": {"actions": [{"id": "e9", "kind": "click", "label": "超" * 40}]}}
    ranked = chat._candidate_ranked(state, {"probabilities": {"e9": 0.9}}, "e9", limit=1)
    assert len(ranked[0][0]) <= 26 and ranked[0][0].endswith("…")


def test_format_decision_line_includes_timing_and_tokens():
    ranked = chat._candidate_ranked(
        _state(), {"probabilities": {"e2": 0.41, "e1": 0.12, "DONE": 0.08}}, "e2"
    )
    line = tui.format_decision(ranked, decision_ms=1830, exec_ms=520, tokens=812, passes=2)
    assert line.startswith("d1 → click「More information」 41%")
    assert "决策 1.8s" in line and "执行 0.5s" in line
    assert "812 tok" in line and "2 轮打分" in line


def test_format_decision_skips_zero_and_unknown_meta():
    line = tui.format_decision([("DONE", 1.0)], decision_ms=0, exec_ms=None, tokens=None, passes=1)
    assert line == "d1 → DONE 100%"


def test_decision_line_prints_one_plain_line(capsys):
    tui.decision_line([("DONE", 0.73), ("CONTINUE", 0.27)], decision_ms=790, tokens=607)
    out = capsys.readouterr().out
    body = [line for line in out.splitlines() if line.strip()]
    assert len(body) == 1 and "DONE 73%" in body[0] and "决策 0.8s" in body[0]


# ------------------------------------------------- llama-server errors ----

def _http_error(code, body):
    return urllib.error.HTTPError("http://x/v1/systemone", code, "err", {}, None) if False else None


def test_request_surfaces_server_error_body():
    def boom(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 400,
            json.dumps({"error": {"message": "exceeds the available context size (8192 tokens)"}}),
            {}, None,
        )

    with mock.patch.object(urllib.request, "urlopen", boom):
        try:
            lb._request("http://x", "/v1/systemone", {}, 5)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as error:
            message = str(error)
            assert "HTTP 400" in message and "exceeds the available context size" in message


def test_is_context_error_matches_only_context_failures():
    assert lb._is_context_error("llama-server HTTP 400: exceeds the available context size (n_ctx)")
    assert not lb._is_context_error("llama-server HTTP 500: invalid chat template")


def test_compact_state_keeps_head_and_tail():
    content = (
        "POLICY...\nGOAL: task\nPAGE:\n" + "页面文本" * 900
        + "\nRECENT ACTIONS:\nNone\nCANDIDATES:\nA: CLICK [1] 链接\nB: DONE: done"
    )
    compact = lb._compact_state(content)
    assert len(compact) < len(content) * 0.45
    assert compact.startswith("POLICY")
    assert compact.rstrip().endswith("B: DONE: done")
    assert "truncated" in compact
    assert lb._compact_state("short") == "short"


def test_systemone_retries_once_with_compacted_state():
    calls = []

    def fake_request(url, path, payload, timeout):
        calls.append(payload["state"])
        if len(calls) == 1:
            raise RuntimeError("llama-server HTTP 400: exceeds the available context size (8192 tokens)")
        return {"answers": {"decision": {"probabilities": {"A": 0.7, "B": 0.3}, "choice": "A"}}}

    model = lb.LlamaCppModel.__new__(lb.LlamaCppModel)
    model.url = "http://x"
    model.labels = ["A", "B"]
    content = "HEAD\nPAGE:\n" + "x" * 4000 + "\nCANDIDATES:\nA: CLICK [1] link\nB: DONE: done"
    with mock.patch.object(lb, "_request", fake_request):
        answer = model._systemone(content, {"type": "choice", "criteria": {"A": "x", "B": "y"}}, purpose="decision")
    assert answer["choice"] == "A"
    assert len(calls) == 2
    assert len(calls[1]) < len(calls[0]) * 0.45 and "truncated" in calls[1]


def test_systemone_propagates_non_context_errors():
    def fake_request(url, path, payload, timeout):
        raise RuntimeError("llama-server HTTP 500: template error")

    model = lb.LlamaCppModel.__new__(lb.LlamaCppModel)
    model.url = "http://x"
    with mock.patch.object(lb, "_request", fake_request):
        try:
            model._systemone("state", {"type": "choice", "criteria": {}}, purpose="decision")
            raise AssertionError("expected RuntimeError")
        except RuntimeError as error:
            assert "template error" in str(error)
