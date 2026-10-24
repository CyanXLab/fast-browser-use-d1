"""Claude-Code-style TUI layer for `fbu chat`.

Renders the banner, step lines, and LLM answers with `rich` when available
(panels, markdown, spinner) and degrades to clean plain-text output otherwise,
so the chat loop works on any terminal (including legacy Windows consoles
after the UTF-8 reconfigure done in chat.py).

Presentation only: no browser/agent logic lives here.
"""

import sys

try:
    from rich.box import ROUNDED
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only without rich installed
    HAS_RICH = False

ACCENT = "bright_cyan"
MUTED = "grey58"
GOOD = "green"
BAD = "red"
WARN = "yellow"

_console = None


def console():
    """Shared rich console (force terminal colors even when piped on Windows)."""
    global _console
    if _console is None and HAS_RICH:
        _console = Console(force_terminal=sys.stdout.isatty() or sys.platform == "win32", highlight=False)
    return _console


def utf8_console():
    """Windows consoles default to GBK; force UTF-8 output with safe fallbacks."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# ---------------------------------------------------------------- banner --

def banner(version, model_line, browser_line, llm_line):
    if console():
        table = Table.grid(padding=(0, 2))
        table.add_column(style=f"bold {ACCENT}", justify="right")
        table.add_column(style="bold")
        table.add_row("⚡ fbu", "Fast Browser Use — interactive chat")
        table.add_row("🧠 model", model_line)
        table.add_row("🌐 browser", browser_line)
        table.add_row("🤝 llm", llm_line)
        console().print(
            Panel(
                table,
                border_style=ACCENT,
                box=ROUNDED,
                padding=(0, 2),
                subtitle=f"[{MUTED}]{version} · /help 查看命令[/]",
                subtitle_align="right",
            )
        )
        return
    print("=" * 64)
    print(f" Fast Browser Use — interactive chat ({version})")
    print(f" model  : {model_line}")
    print(f" browser: {browser_line}")
    print(f" llm    : {llm_line}")
    print(" Type /help for commands.")
    print("=" * 64)


def help_panel(commands):
    """`commands` is a list of (usage, description) tuples."""
    if console():
        table = Table.grid(padding=(0, 3))
        table.add_column(style=f"bold {ACCENT}", no_wrap=True)
        table.add_column()
        for usage, description in commands:
            table.add_row(usage, description)
        console().print(Panel(table, title="命令 / commands", border_style=ACCENT, box=ROUNDED, padding=(0, 2)))
        return
    print("\n".join(f"  {usage:<22} {description}" for usage, description in commands))


# ----------------------------------------------------------------- input --

def prompt(label="❯ "):
    """Read one line with a colored prompt; returns '' on EOF/Ctrl+C."""
    try:
        if console():
            return console().input(f"[bold {ACCENT}]{label}[/]").strip()
        return input(label).strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"


# ----------------------------------------------------------------- steps --

def task_line(text, url):
    if console():
        console().print(Text.assemble(("● ", ACCENT), (text, "bold"), ("  →  ", MUTED), (url, f"{MUTED} underline")))
        return
    print(f"● {text}\n  → {url}")


def step_line(number, action, confidence=None, elapsed_ms=None, status="ready"):
    """One executed agent step, Claude-Code transcript style."""
    detail = f"conf {confidence:.2f}" if isinstance(confidence, (int, float)) else (confidence or "")
    seconds = f"{elapsed_ms / 1000:.1f}s" if isinstance(elapsed_ms, (int, float)) else ""
    color = GOOD if status in {"ready", "predicted"} else (WARN if status == "blocked" else ACCENT)
    meta_text = " · ".join(x for x in (detail, seconds) if x)
    if console():
        head = Text(f"⏺ {number:>2}", style=color)
        body = Text(action, style="bold")
        meta = Text(meta_text, style=MUTED)
        parts = [head, "  ", body]
        if meta_text:
            parts += ["   ", meta]
        console().print(Text.assemble(*parts))
        return
    print(f"  [{number:>2}] {action}" + (f"  ({meta_text})" if meta_text else ""))


def llm_step_line(number, purpose="consult remote LLM", confidence=None, elapsed_ms=None):
    step_line(number, purpose, confidence=confidence, elapsed_ms=elapsed_ms)


# -------------------------------------------------------------- decisions --

def _decision_meta_bits(*, decision_ms=None, exec_ms=None, tokens=None, passes=None):
    """Human-readable timing/cost pieces shared by the rich and plain paths."""
    bits = []
    if isinstance(decision_ms, (int, float)) and decision_ms > 0:
        bits.append(f"决策 {decision_ms / 1000:.1f}s")
    if isinstance(exec_ms, (int, float)) and exec_ms > 0:
        bits.append(f"执行 {exec_ms / 1000:.1f}s")
    if isinstance(tokens, (int, float)) and tokens > 0:
        bits.append(f"{int(tokens)} tok")
    if isinstance(passes, (int, float)) and passes > 1:
        bits.append(f"{int(passes)} 轮打分")
    return bits


def format_decision(ranked, **meta):
    """Plain text of one d1 decision line (used by tests and the fallback path)."""
    parts = [f"{label} {prob * 100:.0f}%" for label, prob in ranked]
    line = "d1 → " + " · ".join(parts)
    bits = _decision_meta_bits(**meta)
    if bits:
        line += "   " + " · ".join(bits)
    return line


def decision_line(ranked, **meta):
    """One dim line revealing how d1-3B scored this step, without flooding.

    `ranked` is [(label, probability), ...] with the executed candidate first;
    meta carries decision_ms / exec_ms / tokens / passes. Renders a single
    indented transcript line under the step.
    """
    if not ranked:
        return
    plain = format_decision(ranked, **meta)
    if console():
        line = Text("   │ ", style=MUTED)
        line.append(f"{ranked[0][0]} {ranked[0][1] * 100:.0f}%", style="bold")
        for label, prob in ranked[1:]:
            line.append(f" · {label} {prob * 100:.0f}%", style=MUTED)
        bits = _decision_meta_bits(**meta)
        if bits:
            line.append("   " + " · ".join(bits), style=MUTED)
        console().print(line)
        return
    print("   │ " + plain)


def plan_panel(steps):
    """Show the decomposed subgoal plan (like a Claude Code task list)."""
    if console():
        table = Table.grid(padding=(0, 2))
        table.add_column(style=f"bold {ACCENT}", justify="right", no_wrap=True)
        table.add_column()
        for index, step in enumerate(steps, 1):
            table.add_row(f"{index}", str(step)[:160])
        console().print(Panel(table, title=f"📋 计划 · {len(steps)} 步", border_style=ACCENT, box=ROUNDED))
        return
    print(f"  [plan: {len(steps)} steps]")
    for index, step in enumerate(steps, 1):
        print(f"    {index}. {step[:140]}")


def milestone(text):
    """One completed subgoal."""
    if console():
        console().print(f"[bold {GOOD}]✓ 子目标完成[/] {text}")
    else:
        print(f"  ✓ subgoal done: {text}")


def answer_panel(text, title="✦ LLM", subtitle=None):
    """Render an LLM answer as markdown inside a panel."""
    if console():
        body = Markdown(text) if text and text.strip() else Text("(empty answer)", style=MUTED)
        console().print(
            Panel(body, title=f"[bold {ACCENT}]{title}[/]", border_style=ACCENT, box=ROUNDED,
                  subtitle=subtitle, subtitle_align="right", padding=(0, 1))
        )
        return
    print(f"── {title} " + "─" * 40)
    print(text or "(empty answer)")


def session_turns(turns):
    """Show the remembered session history (what the LLM can see)."""
    if not turns:
        print("会话历史为空（这是第一轮）。" if console() else "no history yet")
        return
    if console():
        table = Table.grid(padding=(0, 2))
        table.add_column(style=f"bold {ACCENT}", justify="right", no_wrap=True)
        table.add_column()
        for index, turn in enumerate(turns, 1):
            table.add_row(f"#{index}", str(turn.get("task", ""))[:100])
            if turn.get("url"):
                table.add_row("", Text(str(turn["url"])[:100], style=MUTED))
            if turn.get("answer"):
                table.add_row("", Text("✦ " + str(turn["answer"]).replace("\n", " ")[:120], style=MUTED))
        console().print(Panel(table, title=f"会话历史 · {len(turns)} 轮（LLM 可见）", border_style=ACCENT, box=ROUNDED))
        return
    for index, turn in enumerate(turns, 1):
        print(f"  #{index} {str(turn.get('task', ''))[:100]}")
        if turn.get("url"):
            print(f"      url: {turn['url']}")
        if turn.get("answer"):
            print(f"      ✦ {str(turn['answer'])[:120]}")


# ---------------------------------------------------------------- notices --

def info(message):
    if console():
        console().print(f"[{MUTED}]· {message}[/]")
    else:
        print(f"· {message}")


def success(message):
    if console():
        console().print(f"[bold {GOOD}]✔[/] {message}")
    else:
        print(f"[ok] {message}")


def error(message):
    if console():
        console().print(f"[bold {BAD}]✘ {message}[/]")
    else:
        print(f"[error] {message}")


def warn(message):
    if console():
        console().print(f"[bold {WARN}]! {message}[/]")
    else:
        print(f"[warn] {message}")


def spinner(text):
    """A status spinner context manager; no-op when rich is missing."""
    con = console()
    if con:
        return con.status(f"[bold {ACCENT}]{text}[/]", spinner="dots")
    import contextlib

    @contextlib.contextmanager
    def _null():
        yield

    return _null()


def bye():
    if console():
        console().print(f"[{MUTED}]👋 bye.[/]")
    else:
        print("bye.")
