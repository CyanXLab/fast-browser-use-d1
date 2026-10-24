"""CLI setup and downloads must not start browser tasks or model inference."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fast_browser_use import cli


@pytest.mark.parametrize("returncode", [0, 7])
def test_install_browser_uses_own_environment_and_preserves_failure(monkeypatch, returncode):
    install = Mock(return_value=SimpleNamespace(returncode=returncode))
    model = Mock(side_effect=AssertionError("Browser installation must not load model weights"))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli.subprocess, "run", install)
    monkeypatch.setattr("fast_browser_use.model.get_model", model)
    monkeypatch.setattr(sys, "argv", ["fbu", "install-browser"])
    if returncode:
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == returncode
    else:
        cli.main()
    install.assert_called_once_with(
        [sys.executable, "-m", "playwright", "install", "chromium"], check=False,
    )
    model.assert_not_called()


def test_bare_invocation_does_not_start_the_inspector(monkeypatch):
    serve = Mock()
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.demo.serve", serve)
    monkeypatch.setattr(sys, "argv", ["fbu"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    serve.assert_not_called()
    assert exit_info.value.code not in (0, None)


def test_explicit_serve_starts_the_inspector(monkeypatch):
    serve = Mock()
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.demo.serve", serve)
    monkeypatch.setattr(sys, "argv", ["fbu", "serve", "--port", "9999"])
    cli.main()
    serve.assert_called_once_with(9999)


def test_modelscope_downloads_d1_gguf_directly(monkeypatch, tmp_path):
    download = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli.subprocess, "run", download)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fbu", "download", "--source", "modelscope", "--output", str(tmp_path)],
    )
    cli.main()
    url = download.call_args.args[0][-1]
    assert url.startswith("https://modelscope.cn/models/LiquidAI/d1-3B-GGUF/resolve/master/")
    assert url.endswith("d1-3B-Q4_K_M.gguf")
    assert download.call_args.args[0][-2] == str(tmp_path / "d1-3B-Q4_K_M.gguf")


def test_modelscope_supports_higher_quantizations(monkeypatch, tmp_path):
    download = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli.subprocess, "run", download)
    monkeypatch.setattr(
        sys, "argv", ["fbu", "download", "--source", "modelscope", "--model", "d1-q8", "--output", str(tmp_path)],
    )
    cli.main()
    assert download.call_args.args[0][-1].endswith("d1-3B-Q8_0.gguf")


@pytest.mark.parametrize("arguments", [
    ["--url", "https://example.test", "--goal", "Find a result"],
    ["--goal", "Find a result", "--expect-title", "Result"],
    ["--scenario", "research", "--url", "https://example.test", "--goal", "Find", "--expect-title", "Result"],
    ["--scenario", "research", "--expect-title", "Result"],
])
def test_custom_recording_requires_goal_url_and_assertions_before_loading(monkeypatch, arguments):
    record = Mock()
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.recording.record", record)
    monkeypatch.setattr(sys, "argv", ["fbu", "record", *arguments])
    with pytest.raises(SystemExit):
        cli.main()
    record.assert_not_called()


def test_custom_recording_forwards_outcome_separately(monkeypatch):
    record = Mock()
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.recording.record", record)
    monkeypatch.setattr(sys, "argv", [
        "fbu", "record", "--url", "https://example.test", "--goal", "Find a result",
        "--expect-title", "Result", "--expect-text", "First", "--expect-text", "Second",
    ])
    cli.main()
    record.assert_called_once_with(
        None, scenario=None, url="https://example.test", goal="Find a result",
        expected={"url": None, "title": "Result", "text": ["First", "Second"]},
    )


@pytest.mark.parametrize("title, passed", [("Result", True), ("Still loading", False)])
def test_run_independently_rejects_false_done_and_retains_trace(monkeypatch, tmp_path, title, passed):
    import json

    class FakeAgent:
        def __init__(self, url, goal):
            assert (url, goal) == ("https://example.test", "Find a result")
            self.state = {"status": "done"}
            self.browser = SimpleNamespace(evaluate=Mock(return_value={
                "url": "https://example.test", "title": title, "text": "",
            }))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self):
            return iter(())

        def snapshot(self):
            return dict(self.state)

    trace = tmp_path / "result.json"
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr("fast_browser_use.agent.Agent", FakeAgent)
    monkeypatch.setattr("fast_browser_use.model.get_model", Mock())
    monkeypatch.setattr(sys, "argv", [
        "fbu", "run", "https://example.test", "--goal", "Find a result",
        "--expect-title", "Result", "--trace", str(trace),
    ])
    if passed:
        cli.main()
    else:
        with pytest.raises(SystemExit, match="assertions failed"):
            cli.main()
    saved = json.loads(trace.read_text())
    assert saved["status"] == "done" and saved["verification"]["passed"] == passed


@pytest.mark.parametrize("model_arg", [None, "d1", "d1-3b"])
def test_download_defaults_to_d1_gguf_without_conversion(monkeypatch, tmp_path, model_arg):
    from fast_browser_use.model import DEFAULT_DECISION_MODEL

    monkeypatch.delenv("FBU_MODEL", raising=False)

    download = Mock(return_value="/cached/d1.gguf")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    arguments = ["fbu", "download", "--source", "huggingface", "--output", str(tmp_path)]
    if model_arg:
        arguments += ["--model", model_arg]
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    download.assert_called_once_with(
        DEFAULT_DECISION_MODEL, "d1-3B-Q4_K_M.gguf", local_dir=str(tmp_path),
    )


def test_install_browser_can_install_linux_dependencies(monkeypatch):
    install = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli.subprocess, "run", install)
    monkeypatch.setattr(sys, "argv", ["fbu", "install-browser", "--with-deps"])
    cli.main()
    install.assert_called_once_with(
        [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=False,
    )


def test_torch_download_uses_original_pinned_weights_without_inference(monkeypatch):
    from fast_browser_use.model import TORCH_MODEL, TORCH_REVISION

    monkeypatch.delenv("FBU_MODEL", raising=False)
    download = Mock(return_value="/cached/qwen")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", ["fbu", "download", "--backend", "torch", "--source", "huggingface"])
    cli.main()
    download.assert_called_once_with(
        TORCH_MODEL, revision=TORCH_REVISION, local_dir=None,
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
    )


@pytest.mark.parametrize("source", ["mlx"])
def test_removed_sources_are_rejected_before_downloading(monkeypatch, source):
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", ["fbu", "download", "--backend", "torch", "--source", source])
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.parametrize("model_arg,expected_model,expected_rev", [
    ("35b", "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754"),
    ("35b-a3b", "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754"),
    ("Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754"),
    ("9b", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
])
def test_torch_download_supports_selectable_models(monkeypatch, model_arg, expected_model, expected_rev):
    monkeypatch.delenv("FBU_MODEL", raising=False)
    download = Mock(return_value="/cached/qwen")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", [
        "fbu", "download", "--backend", "torch", "--source", "huggingface", "--model", model_arg,
    ])
    cli.main()
    download.assert_called_once_with(
        expected_model, revision=expected_rev, local_dir=None,
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
    )


def test_torch_download_rejects_unsupported_model(monkeypatch):
    monkeypatch.delenv("FBU_MODEL", raising=False)
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", [
        "fbu", "download", "--backend", "torch", "--source", "huggingface", "--model", "unsupported-model",
    ])
    with pytest.raises(SystemExit):
        cli.main()

