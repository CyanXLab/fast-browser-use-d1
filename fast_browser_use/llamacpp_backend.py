"""llama.cpp CPU inference backend for the LiquidAI d1 decision model family.

d1-3B runs through `llama-server` (built with decision-model support) and exposes
the native `/v1/systemone` endpoint: one forward pass answers named, typed
questions over a state with calibrated probabilities and zero generated tokens.
Field text and optional checklists are answered through the OpenAI-compatible
chat endpoint with greedy decoding.

All inference is 100% local CPU; no cloud API is contacted.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

from .model import candidate_codes

DEFAULT_LLAMACPP_URL = "http://127.0.0.1:8765"
DEFAULT_GGUF = "d1-3B-Q4_K_M.gguf"
STATE_DIR = Path(os.environ.get("FBU_STATE_DIR", Path.home() / ".cache" / "fast-browser-use"))
ACTION_INSTRUCTION = (
    "Which candidate code is the correct next browser action for the subgoal? "
    "Candidate codes and their descriptions are listed in the state. "
    "If the subgoal is already satisfied by the current page, choose DONE. "
    "Answer with one code."
)
COMPLETION_INSTRUCTION = (
    "The browser subgoal is fully accomplished only when its final outcome is confirmed "
    "in the observed state: a visible confirmation message, the requested page being open, "
    "or the requested save, submit or search action already executed with its effect observed. "
    "Correctly filled form fields are preparation, not completion. Is the subgoal fully "
    "accomplished right now? Note: a filled field with search results visible below it means "
    "the search WAS submitted and IS complete."
)


def _request(base_url, path, payload, timeout):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # Surface the server's own error message (context overflow, template
        # issues, ...) instead of a bare "HTTP Error 500".
        try:
            detail = error.read().decode("utf-8", "replace").strip()
        except (OSError, ValueError):
            detail = ""
        message = detail[:400] or str(error)
        raise RuntimeError(f"llama-server HTTP {error.code} on {path}: {message}") from error


def _is_context_error(message):
    lowered = str(message).lower()
    return any(hint in lowered for hint in ("context", "exceed", "too long", "n_ctx", "token"))


def _compact_state(content, keep_head=0.15, keep_tail=0.25):
    """Shrink a scored state so it fits the context window.

    Keeps the policy head and the candidates tail (the label-token decision
    mostly needs the candidate table), drops the long page text in between.
    """
    if len(content) < 800:
        return content
    head, tail = int(len(content) * keep_head), int(len(content) * keep_tail)
    marker = "\n...[page text truncated to fit the context window]...\n"
    return content[:head] + marker + content[-tail:]


def _healthy(base_url, timeout=2):
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _extract_options(content):
    """Parse the candidate description lines (A: ..., B: ...) out of a scored prompt."""
    options = {}
    for match in re.finditer(r"^([A-Z]{1,2}): (.+)$", content, flags=re.MULTILINE):
        options.setdefault(match.group(1), match.group(2)[:400])
    return options


class LlamaCppModel:
    """Decision engine backed by a llama.cpp server running a d1 decision model."""

    backend = "llamacpp"
    # d1 calibrates the explicit "is the subgoal accomplished?" question better than
    # competing DONE against actions in a single choice; prefer the binary check.
    prefer_binary_completion = True

    def __init__(self, path=None):
        self.url = os.environ.get("FBU_LLAMACPP_URL", DEFAULT_LLAMACPP_URL).rstrip("/")
        self.name = "LiquidAI/d1-3B"
        self.revision = None
        self.location = "llama.cpp server"
        self.device = "cpu"
        self.dtype = "Q4_K_M"
        self.lock = threading.Lock()
        self.labels, self.label_ids = candidate_codes(_StaticTokenizer(), 256)
        started = time.perf_counter()
        self._ensure_server(path)
        self._load_name()
        self.load_ms = round((time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------------ server

    def _ensure_server(self, path=None):
        if _healthy(self.url):
            return
        model_file = self._resolve_model_file(path)
        binary = self._resolve_binary()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        command = [
            binary, "-m", str(model_file), "-c", os.environ.get("FBU_LLAMACPP_CTX", "16384"),
            "-t", os.environ.get("FBU_LLAMACPP_THREADS") or str(os.cpu_count() or 2),
            "--host", "127.0.0.1", "--port", str(urllib.parse.urlparse(self.url).port or 8765),
        ]
        # KV prefix reuse: consecutive decision steps share the policy/goal
        # prefix, so the server can skip re-evaluating cached prompt tokens.
        # NOTE: for decision models llama.cpp disables the prompt cache by
        # default; --cache-ram opts it in, --cache-reuse matches chunk-wise.
        try:
            cache_reuse = int(os.environ.get("FBU_LLAMACPP_CACHE_REUSE", "256") or 256)
        except ValueError:
            cache_reuse = 256
        try:
            cache_ram = int(os.environ.get("FBU_LLAMACPP_CACHE_RAM", "1024") or 1024)
        except ValueError:
            cache_ram = 1024
        if cache_reuse > 0:
            command += ["--cache-reuse", str(cache_reuse)]
        if cache_ram > 0:
            command += ["--cache-ram", str(cache_ram)]
        log_path = STATE_DIR / "llama-server.log"
        with open(log_path, "ab") as log:
            subprocess.Popen(
                command, stdout=log, stderr=log,
                # POSIX only: detached process group; Windows ignores/handles it differently
                start_new_session=(os.name == "posix"),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                cwd=str(Path(model_file).parent), close_fds=True,
            )
        deadline = time.monotonic() + float(os.environ.get("FBU_LLAMACPP_START_TIMEOUT", "180"))
        while time.monotonic() < deadline:
            if _healthy(self.url):
                return
            time.sleep(1)
        raise RuntimeError(
            f"llama-server did not become healthy at {self.url} within the start timeout. "
            f"Log: {log_path}"
        )

    def _resolve_model_file(self, path=None):
        """FBU_MODEL / explicit path / well-known download locations."""
        candidates = []
        target = str(path or os.environ.get("FBU_MODEL") or "").strip()
        if target:
            candidates.append(Path(target))
        for base in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent.parent, STATE_DIR):
            candidates.extend([base / "models" / DEFAULT_GGUF, base / DEFAULT_GGUF])
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix == ".gguf":
                return candidate
            if candidate.is_dir():
                ggufs = sorted(candidate.glob("*.gguf"))
                if ggufs:
                    return ggufs[0]
        raise RuntimeError(
            f"No d1 GGUF weights found. Run `fbu download` or set FBU_MODEL to "
            f"{DEFAULT_GGUF}. Searched: {[str(c) for c in candidates[:6]]}"
        )

    def _resolve_binary(self):
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        local_tools = Path(__file__).resolve().parent.parent / "tools" / "llama.cpp"
        for candidate in [
            os.environ.get("FBU_LLAMACPP_BIN"),
            shutil.which("llama-server"),
            "build/bin/llama-server",
            "build/bin/llama-server.exe",
            "../llama.cpp/build/bin/llama-server",
            "../llama.cpp/build/bin/llama-server.exe",
            str(local_tools / exe),
            str(STATE_DIR / exe),
        ]:
            if candidate and Path(candidate).expanduser().is_file():
                return str(Path(candidate).expanduser().resolve())
            if candidate and shutil.which(candidate):
                return candidate
        raise RuntimeError(
            "llama-server binary not found. Build llama.cpp (cmake -B build && cmake --build build "
            "--target llama-server) or set FBU_LLAMACPP_BIN to the binary path."
        )

    def _load_name(self):
        try:
            models = _request(self.url, "/v1/models", {}, 10)
            data = models.get("data") or []
            if data:
                self.name = data[0].get("id") or self.name
        except (RuntimeError, urllib.error.URLError, TimeoutError, KeyError, ValueError):
            pass

    # ------------------------------------------------------------------ scoring

    def _systemone(self, state, question, *, purpose="decision"):
        payload = {"state": state, "questions": {purpose: question}}
        timeout = float(os.environ.get("FBU_LLAMACPP_TIMEOUT", "300"))
        try:
            return _request(self.url, "/v1/systemone", payload, timeout)["answers"][purpose]
        except RuntimeError as error:
            if not _is_context_error(error):
                raise
            # One self-healing retry: decision models only need the candidate
            # table and the goal, so drop the long page text and try again.
            payload["state"] = _compact_state(state)
            return _request(self.url, "/v1/systemone", payload, timeout)["answers"][purpose]

    def _score_options(self, content, options, instruction, purpose, *, prompt_tokens=None):
        with self.lock:
            started = time.perf_counter()
            answer = self._systemone(
                content,
                {"type": "choice", "instructions": instruction, "criteria": options},
                purpose="decision",
            )
            probabilities = answer.get("probabilities") or {}
        scores = [float(probabilities.get(label, 0.0)) for label in self.labels[: len(options)]]
        total = sum(scores)
        if total <= 0:
            raise ValueError("llama.cpp returned no probability mass for the candidates")
        scores = [score / total for score in scores]
        return scores, {
            "model": self.name,
            "backend": self.backend,
            "device": self.device,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": {
                "prompt_tokens": prompt_tokens or answer.get("usage", {}).get("input_tokens"),
                "cached_tokens": 0,
                "completion_tokens": 0,
            },
            "cache_hit": False,
            "candidate_count": len(options),
            "confidence": answer.get("confidence"),
            "purpose": purpose,
        }

    def score(self, content, count, *, purpose="action", continuation="", thinking=False, candidates=None):
        if not 1 <= count <= len(self.labels):
            raise ValueError("Invalid candidate count")
        labels = self.labels[:count]
        if purpose == "completion" and count == 2:
            # d1's native yes/no head is far better calibrated for completion than an
            # A/B choice; noul returns P(statement true) in one forward pass.
            return self._score_completion(content, purpose)
        if candidates and len(candidates) == count:
            options = dict(zip(labels, candidates, strict=True))
        else:
            options = _extract_options(content)
            options = {label: options.get(label, f"Candidate {label}") for label in labels}
        instruction = ACTION_INSTRUCTION
        return self._score_options(content, options, instruction, purpose, prompt_tokens=len(content) // 3)

    def _score_completion(self, content, purpose):
        with self.lock:
            started = time.perf_counter()
            answer = self._systemone(
                content,
                {"type": "noul", "instructions": COMPLETION_INSTRUCTION},
                purpose="completion",
            )
            done = float(answer.get("noul", 0.0))
            scores = [1.0 - done, done]
        return scores, {
            "model": self.name,
            "backend": self.backend,
            "device": self.device,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": {"prompt_tokens": len(content) // 3, "cached_tokens": 0, "completion_tokens": 0},
            "cache_hit": False,
            "candidate_count": 2,
            "confidence": answer.get("noul"),
            "purpose": purpose,
        }

    def think_score(self, content, count):
        """d1 decides in a single forward pass; deliberation is not applicable."""
        scores, telemetry = self.score(content, count)
        telemetry["reasoning"] = {
            "model": self.name,
            "note": "d1 is a System-1 decision model; no deliberation channel.",
        }
        return scores, telemetry

    # ---------------------------------------------------------------- generation

    def _chat(self, system, user, max_tokens):
        response = _request(
            self.url,
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                # Reuse the server-side KV cache for the stable system prefix.
                "cache_prompt": True,
            },
            float(os.environ.get("FBU_LLAMACPP_TIMEOUT", "300")),
        )
        return response["choices"][0]["message"]["content"]

    def generate_text(self, context):
        from .model import TEXT_POLICY, parse_field_text

        with self.lock:
            started = time.perf_counter()
        content = self._chat(TEXT_POLICY, json.dumps(context, ensure_ascii=False), 128)
        try:
            value = parse_field_text(content)
        except ValueError:
            value = self._scored_field_text(context)
        telemetry = {
            "model": self.name,
            "backend": self.backend,
            "device": self.device,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "usage": {"prompt_tokens": len(json.dumps(context)) // 3, "completion_tokens": len(content) // 3},
        }
        return value, telemetry

    def _scored_field_text(self, context):
        """System-1 fallback: pick the field value by scoring goal fragments."""
        from .model import extract_field_fragments, parse_field_text

        goal = str(context.get("goal") or context.get("subgoal") or "")
        options = {}
        for fragment in extract_field_fragments(goal):
            options[self.labels[len(options)]] = fragment
            if len(options) >= 8:
                break
        if not options:
            raise ValueError("Local model returned no valid field value; nothing typed.")
        # The refusal option must never leak its own label as a typed value:
        # when the model picks it we fail the step instead of typing meta-text.
        fallback_code = self.labels[len(options)]
        options[fallback_code] = "None of the above; the goal supplies no value for this field."
        answer = self._systemone(
            json.dumps(context, ensure_ascii=False),
            {
                "type": "choice",
                "instructions": (
                    "Which exact text should be typed into the selected browser field? "
                    "Each candidate is a possible field value."
                ),
                "criteria": options,
            },
            purpose="text",
        )
        choice = answer.get("choice")
        if choice is None or choice not in options or choice == fallback_code:
            raise ValueError("Local model returned no valid field value; nothing typed.")
        return parse_field_text(json.dumps({"text": options[choice]}))

    def plan(self, goal, page):
        from .actions import action_space
        from .model import LARGE_PAGE_PLAN_POLICY, PLAN_POLICY, plan_invented_tokens, plan_lost_goal_tokens

        elements, _, _ = action_space(page["actions"])
        labels = [
            {k: e[k] for k in ("label", "role", "options") if k in e} for e in elements if e["role"] != "link"
        ]
        if len(goal) > 200 or len(json.dumps(labels)) > 4000:
            policy, user = LARGE_PAGE_PLAN_POLICY, goal
        else:
            policy = PLAN_POLICY
            user = json.dumps({"goal": goal, "page_title": page["title"], "controls": labels}, ensure_ascii=False)
        with self.lock:
            started = time.perf_counter()
        try:
            content = self._chat(policy, user, 512)
            result = json.loads(re.search(r"\{.*\}", content, flags=re.DOTALL).group(0))
            steps = result.get("steps")
            if (
                isinstance(steps, list)
                and 1 <= len(steps) <= 12
                and all(isinstance(s, str) and 0 < len(s.strip()) <= 500 for s in steps)
                and not plan_lost_goal_tokens(goal, steps)
                and not plan_invented_tokens(goal, steps, [entry.get("label", "") for entry in labels])
            ):
                telemetry = {
                    "model": self.name,
                    "backend": self.backend,
                    "device": self.device,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "usage": {"prompt_tokens": len(user) // 3, "completion_tokens": len(content) // 3},
                    "steps": steps,
                }
                return steps, telemetry
        except (ValueError, AttributeError, RuntimeError, urllib.error.URLError, TimeoutError):
            pass
        # Decision models keep plans minimal: fall back to the raw goal as one subgoal.
        return [goal], {
            "model": self.name,
            "backend": self.backend,
            "device": self.device,
            "latency_ms": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "steps": [goal],
            "fallback": True,
        }


class _StaticTokenizer:
    """The server owns tokenization; labels only need to be stable code strings."""

    def encode(self, text, add_special_tokens=False):
        return [zlib.crc32(text.encode("utf-8"))]
