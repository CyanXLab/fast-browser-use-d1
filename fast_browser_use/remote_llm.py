"""Optional remote-LLM escalation for hard moments, backed by ModelScope
API-Inference (OpenAI-compatible; DeepSeek-V4.1-Flash by default).

The local d1-3B decision model handles every routine step with a single
forward pass. When it is genuinely stuck — low decision confidence, repeated
no-progress actions, or a goal that needs clarification — this module asks a
larger remote LLM for a second opinion, mirroring the planner/fallback role
that browser-use assigns to its LLM.

Everything is configurable through environment variables:

    FBU_LLM_API_BASE    https://api-inference.modelscope.cn/v1
    FBU_LLM_MODEL       deepseek-ai/DeepSeek-V4.1-Flash-1017
    FBU_LLM_API_KEY     ms-...          (change or set empty to disable)
    FBU_LLM_TIMEOUT     60
    FBU_LLM_THINKING    0               (0 = enable_thinking:false / low thinking)
    FBU_LLM_ESCALATION  auto | on | off (default auto)
    FBU_LLM_RESCUE_LIMIT 2              (max stuck-rescues per run)

`auto` escalates only on the triggers below; `on` also lowers the confidence
threshold so the remote LLM double-checks more decisions; `off` disables every
remote call and keeps the run 100% local.
Docs: https://modelscope.cn/docs/model-service/API-Inference/intro
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_API_BASE = "https://api-inference.modelscope.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEFAULT_API_KEY = "ms-d72ed675-91b1-48b1-a495-aed250e1ccf8"
DEFAULT_TIMEOUT = 60.0

DECIDE_INSTRUCTION = (
    "You are the supervisor of a browser-automation agent. A small local decision "
    "model picked its next action with low confidence. Review the observed page "
    "state and the numbered candidate actions, then answer with ONLY the code of "
    "the best candidate (for example: B). If no candidate can advance the goal, "
    "answer DONE when the goal is already satisfied on the page, otherwise answer "
    "UNCERTAIN. Never answer with the BLOCKED code; stopping is the local agent's "
    "own decision. No explanations outside the code."
)

RESCUE_INSTRUCTION = (
    "You are the supervisor of a browser-automation agent that is stuck: its last "
    "actions produced no visible page change. Diagnose the situation from the "
    "observed state and reply with ONLY a JSON object of the exact form:\n"
    '  {"diagnosis": "<one sentence>", "suggested_goal": "<one concrete, visible-'
    'element-oriented browser subgoal to try next>"}\n'
    'If the task itself is ambiguous and you need the user to decide, reply with:\n'
    '  {"need_user": true, "question": "<the question for the user>"}\n'
    "Never invent page elements that are not listed in the observed state. The "
    "agent can ONLY click visible elements and type text into visible fields on "
    "the CURRENT page; it cannot type URLs or navigate directly. Suggested goals "
    "must be achievable on the current page (e.g. search from the current page's "
    "own search box), never 'go to another site' as a separate step."
)

CONSULT_INSTRUCTION = (
    "You assist a browser-automation agent driven by a small local decision model. "
    "The agent chose to consult you because the task needs something beyond plain "
    "page actions: answering a question about the page, summarizing what was done "
    "or what the page shows, or proposing next steps when it is unsure how to "
    "proceed. Ground everything in the observed page state; never invent elements. "
    "This chat is a continuing session: the SESSION HISTORY block lists earlier "
    "user tasks, your earlier answers and the pages reached, so pronouns and "
    "phrases like \"the previous task\", \"it\" or \"continue\" refer to that "
    "history. Reply in exactly this format:\n"
    "ANSWER: <your answer, summary, or brief guidance>\n"
    "OPTIONS: <optional; up to 3 lines, each '- <one concrete next step that "
    "references a visible page element>'; omit the OPTIONS block entirely when "
    "the task only needs an answer or a summary>"
)

RESCUE_INSTRUCTION = (
    "You are the supervisor of a browser-automation agent that is stuck: its last "
    "actions produced no visible page change. This chat is a continuing session: "
    "the SESSION HISTORY block lists earlier user tasks and answers, and the user "
    "may be referring to them (\"continue the previous task\"). Diagnose the "
    "situation from the observed state and reply with ONLY a JSON object of the "
    "exact form:\n"
    '  {"diagnosis": "<one sentence>", "suggested_goal": "<one concrete, visible-'
    'element-oriented browser subgoal to try next>"}\n'
    'If the task itself is ambiguous and you need the user to decide, reply with:\n'
    '  {"need_user": true, "question": "<the question for the user>"}\n'
    "Never invent page elements that are not listed in the observed state. The "
    "agent can ONLY click visible elements and type text into visible fields on "
    "the CURRENT page; it cannot type URLs or navigate directly. Suggested goals "
    "must be achievable on the current page (e.g. search from the current page's "
    "own search box), never 'go to another site' as a separate step."
)

# A first-class candidate the local decision model can score like any action.
# The d1 model itself decides WHEN remote help is worth it, purely from the
# task text: QA tasks, summary requests, or dead ends make this candidate win;
# goals that say "do not use the LLM" make it lose. No hard triggers.
ASK_LLM_CANDIDATE = {
    "id": "ASK_LLM",
    "operation": "ASK_LLM",
    "target": None,
    "description": (
        "ASK_LLM: consult the remote LLM to answer a question about this page, "
        "summarize the result of the task, or propose next-step options when "
        "browser actions alone cannot satisfy the task"
    ),
}


class RemoteLLMError(RuntimeError):
    pass


def format_session_history(turns, *, limit=6, chars=1800):
    """Render earlier chat turns as a compact SESSION HISTORY block.

    `turns` items are dicts like {"task", "answer", "url"} (any field
    optional). Oldest entries are dropped first; the block is truncated to
    `chars` so page text never crowds out the current context.
    """
    if not turns:
        return ""
    lines = []
    for index, turn in list(enumerate(turns, 1))[-limit:]:
        task = str(turn.get("task", "")).strip().replace("\n", " ")[:200]
        url = str(turn.get("url", "")).strip()[:200]
        answer = str(turn.get("answer", "")).strip().replace("\n", " ")
        answer = answer[:400] + ("…" if len(answer) > 400 else "")
        lines.append(f"[{index}] USER TASK: {task or '(navigated)'}")
        if url:
            lines.append(f"    PAGE REACHED: {url}")
        if answer:
            lines.append(f"    LLM ANSWER: {answer}")
    block = "\n".join(lines)
    return block[-chars:]


def _session_block(session_context):
    if not session_context:
        return ""
    return "SESSION HISTORY (earlier turns of this chat):\n" + format_session_history(session_context)


def _cfg(key, default):
    value = os.environ.get(key)
    return value.strip() if value and value.strip() else default


def mode():
    """auto (default) | on | off — off keeps every call fully local."""
    return _cfg("FBU_LLM_ESCALATION", "auto").lower()


def enabled():
    if mode() == "off":
        return False
    # An explicit empty FBU_LLM_API_KEY disables remote calls entirely.
    raw_key = os.environ.get("FBU_LLM_API_KEY")
    if raw_key is not None and not raw_key.strip():
        return False
    return bool(_cfg("FBU_LLM_API_KEY", DEFAULT_API_KEY))


def confidence_threshold():
    try:
        value = float(_cfg("FBU_LLM_CONFIDENCE", "auto" if mode() == "auto" else "0.75"))
    except ValueError:
        value = 0.55 if mode() == "auto" else 0.75
    return value


def chat(messages, *, temperature=0.2, max_tokens=512, timeout=None):
    """One OpenAI-compatible chat completion. Returns (content, telemetry)."""
    base = _cfg("FBU_LLM_API_BASE", DEFAULT_API_BASE).rstrip("/")
    payload = {
        "model": _cfg("FBU_LLM_MODEL", DEFAULT_MODEL),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Hybrid-thinking models: keep latency predictable by defaulting to
        # no/low thinking (FBU_LLM_THINKING=1 enables it).
        "enable_thinking": _cfg("FBU_LLM_THINKING", "0").lower() in {"1", "true", "yes", "on"},
    }
    headers = {"Content-Type": "application/json"}
    api_key = _cfg("FBU_LLM_API_KEY", DEFAULT_API_KEY)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    started = time.perf_counter()
    deadline = float(_cfg("FBU_LLM_TIMEOUT", str(DEFAULT_TIMEOUT)))
    try:
        with urllib.request.urlopen(request, timeout=timeout or deadline) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        raise RemoteLLMError(f"remote LLM request failed: {error}") from error
    latency = round((time.perf_counter() - started) * 1000, 2)
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RemoteLLMError(f"remote LLM returned an unexpected payload: {body}") from error
    usage = body.get("usage") or {}
    return content, {
        "model": payload["model"],
        "backend": "remote-llm",
        "device": "api",
        "latency_ms": latency,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        },
    }


def _parse_json_object(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def decide(prompt, candidates):
    """Second opinion for a low-confidence action choice.

    `prompt` is the exact scored content the local model saw; `candidates` maps
    candidate codes to descriptions. Always returns a record; `choice` is None
    when the remote answer is not one of the listed codes.
    """
    table = "\n".join(f"{code}: {desc}" for code, desc in candidates.items())
    content, telemetry = chat(
        [
            {"role": "system", "content": DECIDE_INSTRUCTION},
            {"role": "user", "content": f"{prompt}\n\nCANDIDATE CODES:\n{table}\nAnswer with one code."},
        ],
        temperature=0.0,
        max_tokens=8,
    )
    answer = content.strip().upper()
    match = re.search(r"\b([A-Z]{1,2})\b", answer)
    code = match.group(1) if match else None
    if code not in candidates:
        code = None
    return {
        "trigger": "low_confidence",
        "choice": code,
        "answer": content.strip()[:120],
        **telemetry,
    }


def consult(task, subgoal, page, recent_actions, local_confidence=None, session_context=None):
    """Open-ended help the local model asked for: page QA, a summary, or options.

    Returns {"answer": str, "options": [str, ...], ...telemetry}. The options,
    when present, are fed back as a refined subgoal so the decision model can
    score real page actions against them (LLM proposes, d1 decides).
    `session_context` is the list of earlier chat turns ({task, answer, url})
    so follow-up tasks like "continue with the previous result" work.
    """
    recent = "\n".join(
        f"{h.get('operation', h['kind'])} {h['action']}" + (f" text={h['text'][:80]!r}" if h.get("text") else "")
        for h in recent_actions[-6:]
    )
    session = _session_block(session_context)
    content, telemetry = chat(
        [
            {"role": "system", "content": CONSULT_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    (f"{session}\n\n" if session else "")
                    + f"USER TASK: {task}\nCURRENT SUBGOAL: {subgoal}\n"
                    f"PAGE TITLE: {page['title']}\nURL: {page['url']}\n"
                    f"ACTIONS EXECUTED SO FAR:\n{recent or 'None'}\n"
                    f"VISIBLE PAGE TEXT (truncated):\n{page['text'][:2400]}"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=512,
    )
    match = re.search(r"ANSWER:\s*(.*?)(?:OPTIONS:\s*(.*))?$", content, flags=re.DOTALL | re.IGNORECASE)
    answer = (match.group(1) if match else content).strip()
    options = []
    if match and match.group(2):
        for line in match.group(2).strip().splitlines():
            line = line.strip().lstrip("-*\u2022 ").strip()
            if line:
                options.append(line[:300])
    record = {"trigger": "ask_llm", "answer": answer[:2000], "options": options[:3], **telemetry}
    if local_confidence is not None:
        record["local_confidence"] = local_confidence
    return record


def rescue_stuck(goal, recent_actions, page, session_context=None):
    """Diagnose a stuck run; may return a revised subgoal or a user question."""
    recent = "\n".join(
        f"{h.get('operation', h['kind'])} {h['action']}"
        + (" (no visible change)" if h.get("page_changed") is False else "")
        for h in recent_actions[-6:]
    )
    session = _session_block(session_context)
    content, telemetry = chat(
        [
            {"role": "system", "content": RESCUE_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    (f"{session}\n\n" if session else "")
                    + f"USER TASK: {goal}\nPAGE TITLE: {page['title']}\nURL: {page['url']}\n"
                    f"RECENT ACTIONS:\n{recent or 'None'}\n"
                    f"VISIBLE PAGE TEXT (truncated):\n{page['text'][:1800]}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=256,
    )
    data = _parse_json_object(content)
    if not data:
        return {
            "trigger": "stuck",
            "diagnosis": content.strip()[:400],
            **telemetry,
        }
    if data.get("need_user"):
        return {"trigger": "ask_user", "question": str(data.get("question", ""))[:400], **telemetry}
    return {
        "trigger": "stuck",
        "diagnosis": str(data.get("diagnosis", ""))[:400],
        "suggested_goal": str(data.get("suggested_goal", ""))[:300],
        **telemetry,
    }
