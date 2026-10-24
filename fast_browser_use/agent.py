"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import os
import time
from pathlib import Path

from .browser import Browser, StalePage
from .model import action_space, choose, field_context, field_text, looks_multistep, make_plan
from .questions import MAX_STEPS


class Agent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=False, video_dir=None, viewport=None,
                 headless=None, session_history=None):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
        self.browser = Browser(url, video_dir=video_dir, viewport=viewport, headless=headless)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
            plan=plan,
            plan_index=0,
            step_history_index=0,
            planned=False,
            plan_calls=[],
            milestones=[],
            decisions=[],
            rejections=[],
            escalations=[],
            answers=[],
            ask_llm_streak=0,
            # Earlier chat turns ({task, answer, url}) injected into remote-LLM
            # prompts so follow-up tasks like "continue the previous one" work.
            session_history=list(session_history or []),
            observations=[],
            text_calls=[],
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    def snapshot(self):
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            before_decisions, before_actions = len(state["decisions"]), len(state["history"])
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage as error:
                # act consumes its decision before checking or dispatching, so
                # retain the prediction index even when state.decision is None.
                state.setdefault("rejections", []).append({
                    "decision_index": before_decisions if len(state["decisions"]) > before_decisions else None,
                    "execution_recorded": len(state["history"]) > before_actions,
                    "reason": str(error),
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                })
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
            # Settle the default full-goal view before spending time on inference.
            plan_mode = os.environ.get("FBU_PLAN", "auto")
            should_plan = plan_mode == "1" or (plan_mode == "auto" and looks_multistep(state["goal"]))
            if not should_plan or "FBU_SETTLE_MS" in os.environ:
                state["page"] = state["browser"].prepare(state["page"], screenshot=self.screenshots)
                if "preparation" in state["page"]:
                    state.setdefault("observations", []).append({
                        **state["page"]["preparation"],
                        "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                    })
            state["decision"] = None
            if state["status"] in {"done", "blocked"}:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if len(state["decisions"]) >= MAX_STEPS * 2:
                raise ValueError("Reached the demo's model-call budget")
            if not state.get("planned", False):
                if should_plan:
                    state["plan"], telemetry = make_plan(state["goal"], state["page"])
                    state["plan_calls"].append(telemetry)
                state["planned"] = True
            focus = state["plan"][state["plan_index"]]
            state["decision"] = choose(
                state["page"], focus, state["history"][state.get("step_history_index", 0) :], task=state["goal"]
            )
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            selected = decision["choice"]
            if selected == "ASK_LLM":
                # The decision model itself asked for remote help (page QA,
                # a summary, or next-step options). No browser action here.
                record = self._consult(decision)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                state["history"].append(
                    {
                        "step": len(state["history"]) + 1,
                        "action": "consult remote LLM",
                        "kind": "ask_llm",
                        "choice": "ASK_LLM",
                        "probability": decision["probabilities"].get("ASK_LLM"),
                        "confidence": decision["confidence"],
                        "latency_ms": decision["latency_ms"],
                        "text": record.get("answer"),
                        "text_helper": record.get("model", "remote-llm"),
                        "text_latency_ms": record.get("latency_ms", 0),
                        "operation": "ASK_LLM",
                        "target": None,
                        "page_changed": False,
                        "url": state["page"]["url"],
                        "usage": decision["usage"],
                        "executed_ms": state["elapsed_ms"],
                        "elapsed_ms": state["elapsed_ms"],
                    }
                )
                # Guardrail: cap consecutive consultations with no browser
                # action in between, otherwise option feedback can make the
                # model ask the LLM in a loop instead of acting.
                state["ask_llm_streak"] = state.get("ask_llm_streak", 0) + 1
                try:
                    streak_limit = max(int(os.environ.get("FBU_ASK_LLM_LIMIT", "2") or 2), 1)
                except ValueError:
                    streak_limit = 2
                if state["ask_llm_streak"] > streak_limit:
                    state["status"] = "done" if record.get("answer") else "blocked"
                    if state["status"] == "done" and record.get("answer"):
                        state["summary"] = record["answer"]
                    return self.snapshot()
                if record.get("options"):
                    # LLM proposes, d1 decides: the options become a refined
                    # subgoal that the decision model scores real actions against.
                    state["plan"].append("Focus suggested by the LLM: " + " | ".join(record["options"]))
                    state["plan_index"] = len(state["plan"]) - 1
                    state["step_history_index"] = len(state["history"])
                    state["status"] = "ready"
                else:
                    state["milestones"].append(state["plan"][state["plan_index"]])
                    state["plan_index"] += 1
                    state["step_history_index"] = len(state["history"])
                    state["status"] = "done" if state["plan_index"] >= len(state["plan"]) else "ready"
                if state["status"] == "done" and record.get("answer"):
                    state["summary"] = record["answer"]
                return self.snapshot()
            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                state["status"] = "done" if selected == "DONE" else "blocked"
                if selected == "DONE":
                    state["milestones"].append(state["plan"][state["plan_index"]])
                    state["plan_index"] += 1
                    state["step_history_index"] = len(state["history"])
                    if state["plan_index"] < len(state["plan"]):
                        state["status"] = "ready"
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                raise ValueError(f"Stopped at the {MAX_STEPS}-action demo budget")
            text, helper = None, None
            if action["kind"] == "fill":
                if not state["browser"].fresh(page, action):
                    raise StalePage("Page changed before text generation. Choose again.")
                focus = state.get("plan", [state["goal"]])[state.get("plan_index", 0)]
                context = field_context(focus, action, page, state["history"], task=state["goal"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})
            # Browser.act checks freshness immediately before input, including after text generation.
            state["browser"].act(action, page, text=text)
            self.pending_text = None
            state["ask_llm_streak"] = 0
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Record execution before observing. A stale post-action observation must not erase the action.
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": action["label"],
                    "kind": action["kind"],
                    "choice": selected,
                    "probability": decision["probabilities"][selected],
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "text": text,
                    "text_helper": helper["model"] if helper else None,
                    "text_latency_ms": helper["latency_ms"] if helper else 0,
                    "operation": decision["operation"],
                    "target": decision["target"],
                    "page_changed": None,
                    "url": page["url"],
                    "usage": decision["usage"],
                    "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                    "elapsed_ms": state["elapsed_ms"],
                }
            )
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
            )
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            repeated = state["history"][-3:]
            stuck = len(repeated) == 3 and all(
                h["page_changed"] is False and h["kind"] not in {"wait", "ask_llm"} for h in repeated
            )
            if stuck:
                rescue = self._rescue()
                if rescue and rescue.get("suggested_goal"):
                    # Self-healing: retry under the LLM-revised subgoal instead of stopping.
                    state["plan"].append(rescue["suggested_goal"])
                    state["plan_index"] = len(state["plan"]) - 1
                    state["step_history_index"] = len(state["history"])
                    state["status"] = "ready"
                else:
                    state["status"] = "blocked"
            else:
                state["status"] = "ready"
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def _consult(self, decision):
        """Remote help the decision model asked for via the ASK_LLM candidate.

        Records the answer (page QA / task summary) and any proposed options;
        API failures never crash the run.
        """
        from . import remote_llm

        state = self.state
        subgoal = state["plan"][state["plan_index"]]
        try:
            record = remote_llm.consult(
                state["goal"], subgoal, state["page"], state["history"], decision.get("confidence"),
                session_context=state.get("session_history"),
            )
        except remote_llm.RemoteLLMError as error:
            record = {"trigger": "ask_llm", "answer": None, "options": [],
                      "error": str(error)[:200]}
        kept = ("answer", "options", "error", "model", "latency_ms", "local_confidence")
        state["answers"].append({k: record[k] for k in kept if k in record})
        return record

    def _rescue(self):
        """Remote-LLM help for a stuck run (browser-use-style fallback).

        Returns a rescue record when the remote LLM proposes a revised subgoal;
        ask-user answers are recorded and the run stops blocked so the CLI can
        surface the question. Failures never crash the run.
        """
        from . import remote_llm

        if not remote_llm.enabled():
            return None
        previous = [e for e in self.state.get("escalations", []) if e.get("trigger") in {"stuck", "ask_user"}]
        try:
            limit = int(os.environ.get("FBU_LLM_RESCUE_LIMIT", "2") or 2)
        except ValueError:
            limit = 2
        if len(previous) >= max(limit, 0):
            return None
        try:
            record = remote_llm.rescue_stuck(
                self.state["goal"], self.state["history"], self.state["page"],
                session_context=self.state.get("session_history"),
            )
        except remote_llm.RemoteLLMError as error:
            self.state["escalations"].append({"trigger": "stuck", "error": str(error)[:200]})
            return None
        self.state["escalations"].append(record)
        return record

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            yield self.command("tick")

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
