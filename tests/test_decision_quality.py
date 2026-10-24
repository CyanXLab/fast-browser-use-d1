"""Decision-quality guards calibrated against a real bing.com run.

Real-world failure modes observed on bing: 40+ navigation candidates diffuse
a 3B model's probabilities (conf 0.05-0.14), the model re-clicks no-effect
links (Maps/Privacy Statement) for many steps, and the generated plan loses
or garbles the goal's key tokens. These tests pin the three guardrails.
"""

from fast_browser_use import model


def click(cid, label):
    return {"id": cid, "operation": "CLICK", "target": "1", "description": f"CLICK [1] {label}"}


def bulk_clicks(start, end, labels=None):
    return [click(f"e{i}", (labels or {}).get(f"e{i}", f"Link {i}")) for i in range(start, end)]


# --- focus_candidates -------------------------------------------------------


def _bing_like_page():
    labels = {f"e{i}": name for i, name in enumerate(
        ["Maps", "News", "Video", "Shopping", "Privacy Statement", "Cookies", "Help", "Feedback",
         "About Bing", "Advertise", "Careers", "More", "Settings", "Sign in"], start=2)}
    return [click("e1", "Enter your search term")] + bulk_clicks(2, 40, labels) + [
        {"id": "e40", "operation": "TYPE_TEXT", "target": "1", "description": "TYPE_TEXT [1] Search box"},
        {"id": "DONE", "operation": "DONE", "target": None, "description": "DONE: complete."},
    ]


def test_focus_caps_large_candidate_tables():
    candidates = _bing_like_page()
    focused = model.focus_candidates(candidates, "去bilibili搜索CyanXLab", limit=10)
    assert len(focused) == 10
    ids = {c["id"] for c in focused}
    assert "e40" in ids and "DONE" in ids  # fill and terminal candidates survive


def test_focus_keeps_goal_keyword_and_action_words():
    candidates = _bing_like_page()
    focused = model.focus_candidates(candidates, "搜索CyanXLab", limit=6)
    assert len(focused) == 8  # limit has a sane floor of 8
    ids = {c["id"] for c in focused}
    assert "e40" in ids and "DONE" in ids  # fill + terminal always survive
    descs = [c["description"].lower() for c in focused]
    # tier-2 action-word clicks beat tier-3 nav links: only 4 nav slots left
    assert any("search" in d for d in descs)
    nav = [d for d in descs if "search" not in d and "done" not in d and "sign in" not in d]
    assert len(nav) == 4


def test_focus_under_limit_is_noop():
    candidates = [click("e1", "One"), click("e2", "Two")]
    assert model.focus_candidates(candidates, "anything") is candidates


def test_focus_matches_goal_keywords():
    candidates = bulk_clicks(0, 30, {**{f"e{i}": f"Nav {i}" for i in range(30)}, "e9": "CyanXLab - YouTube"})
    focused = model.focus_candidates(candidates, "搜索CyanXLab", limit=5)
    ids = {c["id"] for c in focused}
    assert "e9" in ids  # keyword-matching link survives the cap
    assert len(ids) == 8  # limit floored to 8


def test_focus_chinese_goal_tokens_match_chinese_labels():
    candidates = bulk_clicks(0, 30, {**{f"e{i}": f"Nav {i}" for i in range(30)}, "e7": "新闻资讯"})
    focused = model.focus_candidates(candidates, "看新闻", limit=5)
    assert any(c["id"] == "e7" for c in focused)


# --- demote_repeats ---------------------------------------------------------


def _history(entries):
    return [
        {"choice": cid, "kind": "click", "page_changed": changed, "operation": "CLICK", "action": cid}
        for cid, changed in entries
    ]


def test_repeated_no_effect_clicks_lose_mass():
    candidates = [click("maps", "Maps"), click("search", "Search box"), click("news", "News")]
    scores = [0.6, 0.3, 0.1]
    history = _history([("maps", False), ("news", False)])
    adjusted = model.demote_repeats(candidates, scores, history)
    assert adjusted is not None
    probs = dict(zip([c["id"] for c in candidates], adjusted, strict=True))
    assert probs["search"] > 0.5  # mass flows to the untried candidate
    assert abs(sum(adjusted) - 1.0) < 1e-9


def test_repeats_guard_ignores_effective_actions():
    candidates = [click("maps", "Maps"), click("go", "Go")]
    scores = [0.5, 0.5]
    history = _history([("maps", True)])  # page DID change: not a repeat
    assert model.demote_repeats(candidates, scores, history) is None


def test_repeats_guard_skips_when_everything_was_tried():
    candidates = [click("a", "A"), click("b", "B")]
    scores = [0.5, 0.5]
    history = _history([("a", False), ("b", False)])
    assert model.demote_repeats(candidates, scores, history) is None


def test_repeats_guard_never_penalizes_fills():
    candidates = [
        {"id": "box", "operation": "TYPE_TEXT", "target": "1", "description": "TYPE_TEXT [1] Search box"},
        click("go", "Go"),
    ]
    scores = [0.4, 0.6]
    history = _history([("box", False), ("box", False)])
    adjusted = model.demote_repeats(candidates, scores, history)
    probs = dict(zip([c["id"] for c in candidates], adjusted, strict=True))
    assert probs["box"] == 0.4  # retrying a fill is legitimate


# --- plan fidelity ----------------------------------------------------------


def test_plan_that_drops_key_tokens_is_rejected():
    goal = "去bilibili搜索CyanXLab，然后给我一个信息画像"
    assert model.plan_lost_goal_tokens(goal, ["Click the search box", "Type the name", "Click Search"])


def test_plan_with_garbled_but_present_token_passes():
    # d1 once planned "Type CyanXLabLab"; the fill path types the clean goal
    # value anyway, so a garbled-but-present token must not force a fallback.
    goal = "去bilibili搜索CyanXLab"
    assert not model.plan_lost_goal_tokens(goal, ["Click Enter your search term", "Type CyanXLabLab"])


def test_plan_check_ignores_goals_without_latin_tokens():
    assert not model.plan_lost_goal_tokens("打开设个网站的首页", ["Click 首页"])
