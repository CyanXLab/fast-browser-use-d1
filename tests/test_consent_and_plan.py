"""Consent-wall filtering and plan hallucination guards (Task 10)."""

from fast_browser_use.model import (
    consent_wall_filter,
    focus_candidates,
    plan_invented_tokens,
    plan_lost_goal_tokens,
)


def click(label, id="e1"):
    return {"id": id, "operation": "CLICK", "target": 1, "description": f"CLICK [{id}] {label}"}


def fill(label, id="f1"):
    return {"id": id, "operation": "TYPE_TEXT", "target": 0, "description": f"TYPE_TEXT [{id}] {label}"}


# --------------------------------------------------------------- consent wall


def test_privacy_dropped_while_accept_present():
    candidates = [
        fill("Enter your search term"),
        click("Images", "e2"),
        click("Privacy Statement", "e3"),
        click("Accept", "e4"),
    ]
    kept = consent_wall_filter(candidates, "去bilibili搜索CyanXLab")
    labels = [c["description"] for c in kept]
    assert not any("Privacy Statement" in x for x in labels)
    assert any("Accept" in x for x in labels)


def test_privacy_kept_without_accept_control():
    candidates = [click("Privacy Statement", "e2"), click("News", "e3")]
    assert consent_wall_filter(candidates, "搜索CyanXLab") == candidates


def test_privacy_goal_keeps_everything():
    candidates = [click("Accept", "e2"), click("Privacy Statement", "e3")]
    assert consent_wall_filter(candidates, "打开网站的 Privacy Policy 看看") == candidates


def test_consent_variants_detected():
    for accept in ("Accept all", "Agree", "I understand", "Got it", "同意", "接受"):
        candidates = [click(accept, "e2"), click("Cookie Policy", "e3")]
        kept = consent_wall_filter(candidates, "搜索CyanXLab")
        assert len(kept) == 1, accept


def test_consent_filter_after_focus():
    # Integration: focus_candidates keeps accept (action word); the filter then
    # removes the privacy link so the banner cannot hijack the run.
    candidates = [
        fill("Enter your search term", "f1"),
        click("Privacy Statement", "e1"),
        click("Accept", "e2"),
        click("Images", "e3"),
        click("News", "e4"),
        click("Maps", "e5"),
        click("Video", "e6"),
        click("Docs", "e7"),
        click("About", "e8"),
        click("Careers", "e9"),
        click("Blog", "e10"),
    ]
    focused = focus_candidates(candidates, "搜索CyanXLab", limit=8)
    kept = consent_wall_filter(focused, "搜索CyanXLab")
    assert any("Accept" in c["description"] for c in kept)
    assert not any("Privacy" in c["description"] for c in kept)


def test_cookie_banner_button_not_confused_with_cookie_link():
    candidates = [click("Accept cookies", "e2"), click("Cookie Statement", "e3")]
    kept = consent_wall_filter(candidates, "搜索CyanXLab")
    assert len(kept) == 1 and "Accept" in kept[0]["description"]


# ----------------------------------------------------------- plan hallucination


def test_doubled_goal_token_rejected():
    goal = "去bilibili搜索CyanXLab"
    steps = ["Click Enter your search term", "Type CyanXLabLab", "Click Search button"]
    assert plan_invented_tokens(goal, steps, ["Enter your search term", "Search button"])
    assert not plan_lost_goal_tokens(goal, steps)  # old guard misses it


def test_hallucinated_apps_rejected():
    goal = "去bilibili搜索CyanXLab"
    steps = ["Close Bing", "Open Chrome", "Navigate to Bilibili", "Search for CyanXLab"]
    assert plan_invented_tokens(goal, steps, ["Enter your search term"])


def test_legitimate_plan_passes():
    goal = "去bilibili搜索CyanXLab"
    steps = ["Click Enter your search term", "Type CyanXLab", "Click Search button"]
    labels = ["Enter your search term", "Search button", "Images", "News"]
    assert not plan_invented_tokens(goal, steps, labels)


def test_generic_plan_words_pass():
    goal = "搜索CyanXLab"
    steps = ["Type CyanXLab into the search box and open the first result"]
    assert not plan_invented_tokens(goal, steps, ["Search box"])


def test_substring_of_goal_token_passes():
    goal = "搜索CyanXLab"
    steps = ["Type CyanXLab"]  # exact goal token
    assert not plan_invented_tokens(goal, steps, [])


def test_no_latin_goal_skips_check():
    goal = "在豆瓣上找沙丘的影评"
    steps = ["点击搜索框", "输入沙丘"]
    assert not plan_invented_tokens(goal, steps, [])


def test_page_label_tokens_are_allowed():
    goal = "搜索CyanXLab"
    steps = ["Click Enter your search term term", "Click Search button"]
    labels = ["Enter your search term term", "Search button"]
    assert not plan_invented_tokens(goal, steps, labels)
