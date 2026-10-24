"""Local-only CLI, including an optional d1 GGUF download path from Hugging Face or ModelScope."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .demo import load_environment


def _report_escalations(result):
    """Print LLM answers, the final summary, and escalation notes at the end."""
    for record in result.get("answers") or []:
        if record.get("error"):
            print(f"\n[llm] Consultation failed: {record['error']}")
        elif record.get("answer"):
            print(f"\n[llm] Answer: {record['answer']}")
    for record in result.get("escalations") or []:
        trigger = record.get("trigger", "?")
        if trigger == "ask_user" and record.get("question"):
            print(f"\n[llm] Question for you ({trigger}): {record['question']}")
        elif record.get("diagnosis"):
            print(f"\n[llm] Diagnosis ({trigger}): {record['diagnosis']}")
        elif record.get("error"):
            print(f"\n[llm] Escalation failed ({trigger}): {record['error']}")
    if result.get("summary"):
        print(f"\n[llm] Final summary: {result['summary']}")


def main():
    load_environment()
    parser = argparse.ArgumentParser(
        description="Local d1-3B decision-model browser-use skill. Host agents invoke `fbu run`."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a natural-language goal on a URL (skill entrypoint)")
    run.add_argument("url")
    run.add_argument("--goal", required=True)
    run.add_argument("--trace", default="artifacts/run.json")
    browser = sub.add_parser("install-browser", help="Install Chromium for this fbu environment (no model download)")
    browser.add_argument("--with-deps", action="store_true", help="Also install Linux browser system dependencies")
    download = sub.add_parser("download", help="Download d1-3B GGUF weights (Hugging Face or ModelScope mirror)")
    download.add_argument("--source", choices=["huggingface", "modelscope"], default="modelscope",
                          help="Model host; modelscope is the fast mirror and the default")
    download.add_argument("--output", help="Local download directory (default: ./models)")
    recording = sub.add_parser("record", help="Record and independently verify a task (original timing)")
    recording.add_argument(
        "--scenario", choices=["research", "wikipedia", "flights"]
    )
    recording.add_argument("--url", help="Start URL for a custom task; use with --goal and outcome assertions")
    recording.add_argument("--goal", help="Natural-language goal for a custom task")
    recording.add_argument("--output", help="New directory for raw recording and trace")
    for command in (run, recording):
        command.add_argument("--expect-url", help="Exact final URL to verify independently")
        command.add_argument("--expect-title", help="Exact final title to verify independently")
        command.add_argument("--expect-text", action="append", default=[], help="Required visible text; repeatable")
    web = sub.add_parser("serve", help="Optional loopback inspector for debugging candidate scores")
    web.add_argument("--port", type=int, default=int(os.environ.get("FBU_PORT", "8767")))
    chat = sub.add_parser("chat", help="Interactive chat loop over a visible browser (Claude-Code style)")
    chat.add_argument("--url", default="https://www.bing.com", help="Start URL for the first task")
    chat.add_argument("--headless", action="store_true", help="Hide the browser window (visible by default)")
    for command in (run, recording, download, web, chat):
        command.add_argument("--backend", choices=["auto", "llamacpp", "mlx", "torch"], help="Overrides FBU_BACKEND")
        command.add_argument("--model", help="d1 GGUF path/alias (d1, d1-q8, d1-bf16) or local directory (FBU_MODEL)")
    for command in (run, recording, web):
        command.add_argument("--device", help="PyTorch device: auto, cpu, cuda or cuda:N (FBU_DEVICE)")
        command.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], help="PyTorch dtype")
    args = parser.parse_args()
    for option in ("backend", "device", "dtype", "model"):
        value = getattr(args, option, None)
        if value is not None:
            os.environ[f"FBU_{option.upper()}"] = value
    if args.command in {"run", "record"}:
        expected = {"url": args.expect_url, "title": args.expect_title, "text": args.expect_text}
        has_expectations = args.expect_url is not None or args.expect_title is not None or bool(args.expect_text)
        if has_expectations:
            from .verification import validate_expectations

            try:
                validate_expectations(**expected)
            except ValueError as exc:
                parser.error(str(exc))
    if args.command == "record":
        custom = args.url is not None or args.goal is not None
        if custom and (not args.url or not args.goal or args.scenario or not has_expectations):
            parser.error("Custom recording needs --url, --goal and at least one --expect-*; omit --scenario")
        if has_expectations and not custom:
            parser.error("--expect-* is for custom recordings with --url and --goal")
    if args.command == "install-browser":
        command = [sys.executable, "-m", "playwright", "install"]
        if args.with_deps:
            command.append("--with-deps")
        result = subprocess.run([*command, "chromium"], check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    elif args.command == "download":
        from .model import DEFAULT_DECISION_GGUF, MODELSCOPE_REPO, decision_quant, model_source, resolve_backend

        backend = resolve_backend()
        if args.source == "modelscope":
            quant = decision_quant(args.model)
            filename = DEFAULT_DECISION_GGUF.replace("Q4_K_M", quant)
            output = Path(args.output or "models")
            output.mkdir(parents=True, exist_ok=True)
            destination = output / filename
            if destination.exists() and destination.stat().st_size > 1_000_000:
                print(f"Already downloaded: {destination}")
                print(f"Set FBU_MODEL={destination}")
            else:
                url = f"https://modelscope.cn/models/{MODELSCOPE_REPO}/resolve/master/{filename}"
                print(f"Downloading {filename} from ModelScope ({MODELSCOPE_REPO})...")
                result = subprocess.run(
                    ["curl", "-L", "--retry", "5", "-C", "-", "-o", str(destination), url], check=False,
                )
                if result.returncode:
                    raise SystemExit(result.returncode)
                print(f"Set FBU_MODEL={destination.resolve()}")
        else:
            try:
                name, revision = model_source(backend, args.model)
            except ValueError as exc:
                parser.error(str(exc))
            if backend == "llamacpp" or name.lower().endswith(".gguf") or "d1-3b-gguf" in name.lower():
                # Decision-model GGUF through the Hugging Face hub.
                quant = decision_quant(args.model)
                filename = DEFAULT_DECISION_GGUF.replace("Q4_K_M", quant)
                from huggingface_hub import hf_hub_download

                location = hf_hub_download(name, filename, local_dir=args.output)
                print(f"Set FBU_MODEL={Path(location).resolve()}")
            else:
                from huggingface_hub import snapshot_download

                location = snapshot_download(
                    name, revision=revision,
                    local_dir=args.output, allow_patterns=["*.json", "*.jinja", "*.safetensors"],
                )
                print(f"Set FBU_MODEL={Path(location).resolve()}" if args.output else location)
    elif args.command == "record":
        from .recording import record

        record(
            args.output, scenario=args.scenario,
            url=args.url, goal=args.goal, expected=expected if has_expectations else None,
        )
    elif args.command == "run":
        from .agent import Agent
        from .model import get_model
        from .verification import verify_outcome

        get_model()
        folder = Path(args.trace)
        folder.parent.mkdir(parents=True, exist_ok=True)
        with Agent(args.url, args.goal) as agent:
            try:
                for state in agent.run():
                    last = state["history"][-1] if state["history"] else {}
                    print(state["elapsed_ms"], state["status"], last.get("action", ""), flush=True)
            finally:
                result = agent.snapshot()
                if has_expectations:
                    try:
                        result["verification"] = verify_outcome(agent.browser, **expected)
                    except Exception as exc:
                        result["verification"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
                folder.write_text(json.dumps(result, indent=2))
            if agent.state["status"] != "done":
                raise SystemExit("Agent stopped without reporting completion; inspect the trace")
            if has_expectations and not result["verification"]["passed"]:
                raise SystemExit("Outcome assertions failed; inspect the trace. Do not blindly rerun mutations.")
        _report_escalations(result)
        print("Outcome assertions passed. Trace:" if has_expectations else
              "Agent reports done. Independently verify the outcome. Trace:", folder)
    elif args.command == "serve":
        from .demo import serve

        serve(args.port)
    elif args.command == "chat":
        from .chat import chat as run_chat

        run_chat(start_url=args.url, headless=args.headless)


if __name__ == "__main__":
    main()
