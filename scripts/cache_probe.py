"""One-shot cache investigation: start llama-server verbosely, benchmark, inspect n_past."""

import subprocess
import sys
import time
import urllib.request

ROOT = "/home/z/my-project"
SERVER = f"{ROOT}/llama.cpp/build/bin/llama-server"
MODEL = f"{ROOT}/fast-browser-use/models/d1-3B-Q4_K_M.gguf"
LOG = "/tmp/llama-verbose2.log"
URL = "http://127.0.0.1:8765"


def main():
    subprocess.run(["pkill", "-f", "llama-server"], check=False)
    time.sleep(2)
    log = open(LOG, "wb")
    proc = subprocess.Popen(
        [SERVER, "-m", MODEL, "-c", "8192", "-t", "2", "--host", "127.0.0.1", "--port", "8765",
         "--cache-reuse", "256", "--cache-ram", "1024", "-lv", "5"],
        stdout=log, stderr=log, start_new_session=True, cwd=ROOT,
    )
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(URL + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise SystemExit("server never became healthy")

        sys.path.insert(0, f"{ROOT}/fast-browser-use/scripts")
        import importlib.util
        spec = importlib.util.spec_from_file_location("bench", f"{ROOT}/fast-browser-use/scripts/bench_cache.py")
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        # run main() twice (second pass = everything warm)
        import contextlib
        import io
        for run in range(2):
            print(f"--- run {run + 1} ---")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bench.main()
            print(buf.getvalue())
    finally:
        proc.terminate()
        proc.wait(timeout=15)

    # inspect verbose log for cache decisions
    hits = []
    for line in open(LOG, encoding="utf-8", errors="replace"):
        low = line.lower()
        if any(k in low for k in ("n_past", "cache_hit", "reuse", "full prompt", "prompt cache", "cache ram")):
            hits.append(line.rstrip()[:220])
    print("==== cache-related verbose lines ====")
    for line in hits[-40:]:
        print(line)


if __name__ == "__main__":
    main()
