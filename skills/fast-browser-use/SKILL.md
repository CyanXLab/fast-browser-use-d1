---
name: fast-browser-use
description: Operate web pages with a local model through the fbu CLI on Linux, Windows or macOS (PyTorch CUDA/CPU or Apple Silicon MLX). Use for browser navigation, searches, standard forms and dropdowns when the user wants local inference; accept any starting URL and natural-language goal.
---

# Fast Browser Use

Delegate the browser interaction loop to `fbu run`. It reads visible DOM, offers legal actions to a
local model, generates field text locally, and checks freshness before execution. The host supplies
the user's goal and verifies the result. No cloud inference is used inside the loop.

## Runtime setup

The skill contains instructions; the Python runtime, Chromium and weights are separate.
Check `fbu --help`. If missing, follow the source repository's README installation instructions;
do not assume a same-named PyPI package is this project. `uv tool install` exposes the CLI across
projects; `uv tool update-shell` and restarting the host may be needed for PATH discovery.

Prepare an installed runtime once with `fbu install-browser` and `fbu download`. The supported model is
LiquidAI d1-3B (GGUF Q4_K_M, ~1.7 GB), pinned by default. All platforms run the decision model on
llama.cpp CPU inference through the /v1/systemone endpoint; no GPU is required.
`FBU_BACKEND=auto` selects MLX on Apple Silicon and PyTorch elsewhere. `FBU_MODEL` can point to
matching local weights; remove stale overrides when switching backends. Other architectures/sizes are rejected.

For a Linux server, run `uv sync --locked --extra torch`, `uv run fbu install-browser --with-deps`,
and `uv run fbu download --backend torch`. Run tasks with `--backend torch --device cuda` (or `cpu`).
`cuda:N` selects a visible GPU. CUDA defaults to BF16 when supported, otherwise FP16; CPU uses FP32.
Allow roughly 18 GB for BF16/FP16 weights plus runtime memory, or 36 GB for FP32 weights alone.
No desktop, DISPLAY or Xvfb is needed. Windows uses the same CLI without `--with-deps`.

From a checkout, use `uv run --project /absolute/path/to/fast-browser-use fbu` in place of `fbu` after
`uv sync --locked`. Resolve model and trace paths against the user's working directory.

## Execute an arbitrary goal

```bash
FBU_MODEL=/absolute/path/to/d1-3B-Q4_K_M.gguf \
fbu run 'https://target.example/' \
  --goal 'The user-requested outcome and constraints' \
  --trace artifacts/task.json
```

The default uses the complete goal, one joint action/completion decision and bounded page settling
(`FBU_PLAN=0`, `FBU_REASONING=0`). `FBU_HEADLESS=0` displays the browser. No inspector UI is required.

Pass known end conditions when they can independently establish the requested outcome:

```bash
fbu run 'https://target.example/settings' \
  --goal 'Save workspace preferences with timezone Asia/Singapore and weekly digest enabled.' \
  --expect-title 'Preferences saved' \
  --expect-text 'Timezone: Asia/Singapore.' \
  --expect-text 'Weekly digest: enabled.' \
  --trace artifacts/preferences.json
```

`--expect-url` and `--expect-title` match exactly. Repeat `--expect-text` to require every fragment
in rendered body text. Assertions run on a fresh browser read after execution and are never fed
to the model as an action plan. They prove only the conditions supplied; a generic “Saved” message
alone does not prove field values. Without assertions, exit zero means the model reported DONE.

Read `status`, `verification`, `page`, `history`, `rejections` and `elapsed_ms` from the trace.
A DONE claim or action history alone does not prove success. If built-in assertions cannot establish
the outcome, independently inspect the resulting state before reporting completion.

## Record any task

```bash
FBU_MODEL=/absolute/path/to/d1-3B-Q4_K_M.gguf \
fbu record --url 'https://target.example/' --goal 'The requested outcome' \
  --expect-url 'https://target.example/result' --output artifacts/recordings/task
```

Custom recordings require a URL, goal and at least one outcome assertion. `--scenario` options are
optional development demos, not a supported-sites list. Recordings always use headless Chromium, including when `FBU_HEADLESS=0` is set.
Raw recordings preserve all inference and waits. Preview rendering requires system ffmpeg/ffprobe. From the checkout, `scripts/render_demo.py RECORDING_DIR --name task --max-seconds 10`
creates a labeled accelerated preview, preserves the original video, and records actual task time
and playback speed separately. Only independently verified completed runs can be rendered.

## Execution boundaries

- Supply outcomes and user constraints; the local model chooses steps and generates field values.
  Never replace the loop with host-generated selectors, executable model code, prepared field
  strings or site-specific action plans.
- Never automatically rerun a failed task that may have mutated the site. Reconcile the actual
  state before further authorized action. Fresh observations may reject stale decisions; dispatched
  mutations are recorded before post-action observation.
- Candidate softmax scores are relative preferences, not calibrated correctness probabilities.
- Optional remote-LLM advisor (ModelScope API-Inference, DeepSeek-V4.1-Flash by default):
  the decision model itself decides when to consult it via the first-class ASK_LLM
  candidate (page QA, task summaries, second opinions on low confidence, stuck-run
  rescue). Configure with FBU_LLM_API_BASE / FBU_LLM_MODEL / FBU_LLM_API_KEY /
  FBU_LLM_ESCALATION (auto|on|off) / FBU_LLM_CONFIDENCE; set FBU_LLM_API_KEY=""
  or FBU_LLM_ESCALATION=off for a 100% local run. Answers land in the trace
  `answers`/`summary`; a goal that says not to use the LLM makes d1 ignore it.
- The browser uses a fresh isolated profile. Existing logins are not inherited. V1 lacks nested
  iframe/deep shadow traversal, canvas interaction, uploads and multi-tab orchestration.
- Traces contain page data and generated field values; keep personal traces and credentials out of git.

The runtime is site-independent; reliability is experimental. d1-3B completed the measured
Wikipedia task at a 30.1 s median after optimization (three verified trials). The release examples
also cover simple navigation and a settings form; these do not establish general-web reliability.
Consult `docs/performance.md` in the repository for measurement boundaries and reproducible checks.
