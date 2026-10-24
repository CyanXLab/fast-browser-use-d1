"""Interactive chat loop (Claude-Code style TUI) over the browser agent.

Run `fbu chat` after deployment: type a task in natural language, watch the
visible browser execute it step by step, get the answer/summary rendered as
markdown, then type the next task. The session remembers the last page AND the
conversation turns; every remote-LLM call receives the earlier turns, so
follow-ups like "继续上一个任务" / "把刚才的结果再细化" work naturally.

In-loop commands (slash forms are the canonical ones):
    /open <url>    navigate somewhere without running a task
    /history       show the session turns the LLM can see
    /clear         forget the session history (browser stays where it is)
    /llm           show the remote-LLM configuration status
    /help          show the command list
    /exit /quit    leave the chat (Ctrl+C / Ctrl+D also work)
"""

import json
import os
import re
from pathlib import Path

from . import tui
from .tui import console

DEFAULT_START_URL = "https://www.bing.com"
SESSION_FILE = Path("artifacts/chat-session.json")
TRACE_FILE = Path("artifacts/chat-trace.json")

COMMANDS = [
    ("/open <url>", "只导航到指定页面，不执行任务"),
    ("/history", "查看 LLM 能看到的会话历史轮次"),
    ("/retry", "重试上一条任务"),
    ("/decide", "开关每步 d1-3B 决策细节（候选概率/耗时拆分）"),
    ("/clear", "清空会话历史（含跨启动的持久记忆）"),
    ("/llm", "查看远程 LLM 协作配置状态"),
    ("/help", "显示本命令列表"),
    ("<任务文本>", "执行浏览器任务，例：打开 baidu.com 搜索今日新闻并总结"),
    ("/exit /quit", "退出（Ctrl+C / Ctrl+D 也可以）"),
]


def load_session():
    """Restore the persisted chat memory ({turns, url}) from the last run."""
    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("turns") or [], data.get("url") or DEFAULT_START_URL
    except (OSError, ValueError):
        pass
    return [], None


def save_session(turns, url):
    """Persist chat memory so the next `fbu chat` can continue where this left off."""
    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(
            json.dumps({"turns": turns, "url": url}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _autosave_trace(snapshot):
    """Checkpoint the live run after every step (crash-safe, Claude-Code style)."""
    try:
        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        TRACE_FILE.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError):
        pass


def _llm_status_line():
    from . import remote_llm

    if not remote_llm.enabled():
        return "off（FBU_LLM_ESCALATION=off 或未配置 key）"
    return f"on · {remote_llm._cfg('FBU_LLM_MODEL', remote_llm.DEFAULT_MODEL)} · 触发 {remote_llm.mode()}"


def _model_line():
    from .model import get_model

    model = get_model()
    name = getattr(model, "name", None) or getattr(model, "model_name", None) or type(model).__name__
    location = getattr(model, "location", "") or ""
    dtype = getattr(model, "dtype", "") or ""
    bits = [str(name)]
    if location and location != str(name):
        bits.append(str(location))
    if dtype:
        bits.append(str(dtype))
    return " · ".join(bits)


def _utf8_first():
    """Force UTF-8 before anything renders (Windows GBK consoles)."""
    tui.utf8_console()


def _print_answers(result):
    """Render the LLM answers/summary produced during the task.

    Multiple consultations in one task usually refine the same answer, so
    only the last one gets a full panel; earlier ones collapse to one line.
    """
    answers = [r for r in result.get("answers") or [] if r.get("answer")]
    for record in result.get("answers") or []:
        if record.get("error"):
            tui.error(f"LLM 咨询失败: {record['error']}")
    if len(answers) > 1:
        tui.info(f"LLM 中间回答 x{len(answers) - 1}（已折叠）: " + answers[0]["answer"].replace("\n", " ")[:110] + "…")
    if answers:
        record = answers[-1]
        subtitle = f"[{tui.MUTED}]{record.get('model', 'remote-llm')} · {record.get('latency_ms', 0):.0f}ms[/]"
        tui.answer_panel(record["answer"], subtitle=subtitle)
    for record in result.get("escalations") or []:
        trigger = record.get("trigger", "?")
        if trigger == "ask_user" and record.get("question"):
            tui.warn(f"LLM 需要你决定: {record['question']}")
        elif record.get("diagnosis"):
            tui.info(f"LLM 诊断（{trigger}）: {record['diagnosis']}")
        elif record.get("error"):
            tui.error(f"LLM 升级失败（{trigger}）: {record['error']}")
    if result.get("summary") and result["summary"] not in [a.get("answer") for a in answers]:
        tui.answer_panel(result["summary"], title="✦ 任务总结")


def _session_turn(session_history, goal, result, url):
    """Record one completed turn so later LLM calls can see it."""
    answer = None
    for record in reversed(result.get("answers") or []):
        if record.get("answer"):
            answer = record["answer"]
            break
    if not answer and result.get("summary"):
        answer = result["summary"]
    if not answer:
        answer = f"任务结束，状态 {result.get('status', '?')}，到达页面 {url}"
    session_history.append({"task": goal, "answer": answer, "url": url})
    del session_history[:-12]  # keep the prompt bounded


def _candidate_ranked(state, decision, chosen_id, limit=3):
    """Top d1 candidates as human-readable (label, probability) pairs.

    The executed candidate always appears (appended if it fell outside the
    top-N); labels come from the observed page actions so candidates render as
    e.g. `click「搜索」` instead of raw ids like `e12`.
    """
    probs = decision.get("probabilities") or {}
    labels = {}
    for action in (state.get("page") or {}).get("actions") or []:
        kind = action.get("kind", "?")
        label = action.get("label") or ""
        labels[action["id"]] = f"{kind}「{label}」" if label else kind
    for control in ("DONE", "BLOCKED", "ASK_LLM", "CONTINUE"):
        labels.setdefault(control, control)
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    out, seen = [], set()
    for action_id, prob in ranked:
        seen.add(action_id)
        label = labels.get(action_id, str(action_id))
        out.append((label if len(label) <= 26 else label[:25] + "…", float(prob)))
        if len(out) >= limit:
            break
    if chosen_id and chosen_id not in seen and chosen_id in probs:
        label = labels.get(chosen_id, str(chosen_id))
        out.append((label if len(label) <= 26 else label[:25] + "…", float(probs[chosen_id])))
    return out


def _render_step(step_number, state, last, step_total, decision, show_detail):
    """Print one transcript step: the action line plus (optionally) the d1
    decision line underneath — one extra dim line, never more."""
    usage = last.get("usage") or {}
    if last.get("kind") == "ask_llm":
        tui.llm_step_line(
            step_number, "✦ 咨询远程 LLM…", confidence=last.get("confidence"),
            elapsed_ms=last.get("text_latency_ms"),
        )
        if show_detail and decision:
            tui.decision_line(
                _candidate_ranked(state, decision, last.get("choice")),
                decision_ms=last.get("latency_ms"),
                tokens=usage.get("prompt_tokens"),
            )
        return
    model_ms = last.get("latency_ms") or 0
    text_ms = last.get("text_latency_ms") or 0
    tui.step_line(
        step_number, last.get("action", "(thinking)"),
        confidence=last.get("confidence"), elapsed_ms=step_total, status=state["status"],
    )
    if show_detail and decision:
        tui.decision_line(
            _candidate_ranked(state, decision, last.get("choice")),
            decision_ms=model_ms or None,
            exec_ms=max((step_total or 0) - model_ms - text_ms, 0) or None,
            tokens=usage.get("prompt_tokens"),
            passes=decision.get("scoring_passes"),
        )
        _render_escalation(decision)


def _render_escalation(decision):
    """One dim line when the remote LLM reviewed a low-confidence decision.

    The user otherwise cannot tell whether the supervisor actually fired
    (observed confusion: "ASK_LLM 5%" in the candidate line but no trace of
    the remote verdict). Failures are shown too, so a dead API key surfaces.
    """
    record = decision.get("escalation")
    if not record:
        return
    trigger = record.get("trigger", "low_confidence")
    if record.get("error"):
        tui.warn(f"🤝 远程复核失败（{trigger}）: {record['error']}")
    elif record.get("candidate"):
        local = decision.get("action") or record.get("local_choice") or "?"
        tui.info(
            f"🤝 远程复核（本地 conf {record.get('local_confidence', 0):.2f}）: "
            f"{local} → 改判 {record['candidate']}"
        )
    elif record.get("ignored"):
        tui.info(f"🤝 远程复核未采纳（{record.get('ignored')}），维持本地选择")


def chat(start_url=DEFAULT_START_URL, headless=False):
    _utf8_first()
    from . import remote_llm
    from .agent import Agent

    session_history, saved_url = load_session()
    current_url = saved_url or start_url

    # Warm the llama-server up before drawing the banner so the browser does
    # not sit open while the model loads.
    with tui.spinner("加载 d1-3B 决策模型（首次启动约 10–30 秒）…"):
        model_line = _model_line()

    memory_line = f"{len(session_history)} 轮已恢复（{SESSION_FILE}）" if session_history else "空（首轮）"
    tui.banner(
        "fbu 0.1.0",
        model_line,
        "可见（headed）" if not headless else "隐藏（--headless）",
        _llm_status_line(),
    )
    tui.info(f"持久记忆: {memory_line}")
    if saved_url and session_history:
        tui.info(f"上次停在: {saved_url}")

    last_goal = None
    while True:
        text = tui.prompt()
        if not text:
            continue
        lowered = text.lower()
        if lowered in {"/exit", "/quit", "exit", "quit", "q", "退出"}:
            break

        # ---- slash commands -------------------------------------------
        if lowered in {"/help", "help"}:
            tui.help_panel(COMMANDS)
            continue
        if lowered.startswith("/open ") or lowered.startswith("open "):
            target = text.split(" ", 1)[1].strip()
            current_url = target if re.match(r"^https?://", target) else f"https://{target}"
            tui.info(f"将在下一个任务前打开: {current_url}")
            continue
        if lowered in {"/history", "history"}:
            tui.session_turns(session_history)
            continue
        if lowered in {"/clear", "clear"}:
            session_history.clear()
            save_session(session_history, current_url)
            tui.success("会话历史已清空（含持久记忆文件）。")
            continue
        if lowered == "/llm":
            tui.info(remote_llm._cfg("FBU_LLM_API_BASE", remote_llm.DEFAULT_API_BASE))
            tui.info(_llm_status_line())
            continue
        if lowered in {"/retry", "retry"}:
            if not last_goal:
                tui.warn("还没有可重试的任务。")
                continue
            text = last_goal  # re-enter the task branch below
        if lowered in {"/decide", "decide"}:
            on = os.environ.get("FBU_STEP_DETAIL", "1") != "0"
            os.environ["FBU_STEP_DETAIL"] = "0" if on else "1"
            tui.success("每步 d1 决策细节已" + ("关闭" if on else "开启") + "（下一个任务生效）。")
            continue
        if lowered.startswith("/") and lowered not in {"/retry", "retry"}:
            tui.warn(f"未知命令 {text.split()[0]}，/help 查看可用命令。")
            continue

        # ---- a browser task -------------------------------------------
        last_goal = text
        match = re.search(r"https?://[^\s，。]+", text)
        if match:
            current_url = match.group(0)
        goal = re.sub(r"https?://[^\s，。]+", "", text).strip() or text
        tui.task_line(goal, current_url)
        try:
            with Agent(current_url, goal, headless=headless, session_history=session_history) as agent:
                plan_shown = False
                seen_milestones = 0
                show_detail = os.environ.get("FBU_STEP_DETAIL", "1") != "0"
                with tui.spinner("⚡ d1-3B 决策中…"):
                    step_number = 0
                    prev_elapsed = 0
                    seen_steps = 0
                    for state in agent.run():
                        step_number += 1
                        step_total = max(state["elapsed_ms"] - prev_elapsed, 0)
                        prev_elapsed = state["elapsed_ms"]
                        history_len = len(state["history"])
                        decision = state["decisions"][-1] if state["decisions"] else None
                        if not plan_shown and len(state.get("plan") or []) > 1:
                            tui.plan_panel(state["plan"])
                            plan_shown = True
                        if history_len > seen_steps:
                            # A browser action or consultation was executed.
                            seen_steps = history_len
                            _render_step(step_number, state, state["history"][-1],
                                         step_total or None, decision, show_detail)
                        elif decision and state["status"] in {"done", "blocked"}:
                            # Terminal verdict consumes a decision without a
                            # history entry; print it instead of the old step.
                            op = decision.get("operation")
                            verdict = "DONE（判定子目标完成）" if op == "DONE" else "BLOCKED（判定无法推进）"
                            tui.step_line(step_number, verdict, confidence=decision.get("confidence"),
                                          elapsed_ms=step_total or None, status=state["status"])
                            if show_detail:
                                tui.decision_line(
                                    _candidate_ranked(state, decision, decision.get("choice")),
                                    decision_ms=decision.get("latency_ms"),
                                    tokens=((decision.get("usage") or {}).get("prompt_tokens")),
                                    passes=decision.get("scoring_passes"),
                                )
                        elif len(state.get("milestones") or []) > seen_milestones:
                            pass  # mid-plan DONE: the milestone line below suffices
                        elif state["status"] == "ready" and decision:
                            tui.info("页面已变化，重新观察后再决策")
                        while seen_milestones < len(state.get("milestones") or []):
                            tui.milestone(state["milestones"][seen_milestones])
                            seen_milestones += 1
                        _autosave_trace(state)
                result = agent.snapshot()
                if agent.state.get("page"):
                    current_url = agent.state["page"].get("url", current_url)
            _print_answers(result)
            _session_turn(session_history, goal, result, current_url)
            save_session(session_history, current_url)
            status = result.get("status", "?")
            if status == "done":
                tui.success(f"任务完成 · {current_url}")
            else:
                tui.warn(f"任务未完成（{status}）· {current_url} · 可继续对话补充指令或 /retry")
        except KeyboardInterrupt:
            if console():
                console().print()
            tui.warn("任务已中断，浏览器会话已关闭。继续输入下一个任务即可。")
        except Exception as error:  # keep the chat alive on any task failure
            message = str(error).strip().splitlines()[0][:200] if str(error).strip() else ""
            tui.error(f"任务失败: {type(error).__name__}: {message}")
    save_session(session_history, current_url)
    tui.bye()
