"""Run serial, checkpointed private experiments; stop only owned child processes."""
import argparse
import json
import os
import subprocess
import sys
import time

import requests

from ipindia_corpus.llm_experiment import ROOT, canonical, timestamp
from ipindia_corpus.paths import immutable_write
from tools.reliability_v2 import DATA, DEST
from tools.reliability_v2_runtime import evaluate, fingerprint, session


def run(variants, mode, split, passes):
    client = session()
    for variant in variants:
        if mode == "controlled":
            evaluate(variant, mode, split, passes)
            continue
        # Never terminate a process discovered by its port or an old PID receipt.
        try:
            client.get("http://127.0.0.1:8001/health", timeout=2)
        except requests.ConnectionError:
            pass
        else:
            raise ValueError("private_port_in_use; stop its verified owner before matrix")
        log_id = timestamp().replace(":", "").replace("+", "_")
        output = ROOT / DEST / f"{variant}-{log_id}.log"
        with output.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "tools.reliability_v2_runtime", "serve-answer", "--variant", variant],
                                       cwd=ROOT, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                for _ in range(120):
                    if process.poll() is not None:
                        raise ValueError("preview_process_exited; inspect " + str(output))
                    try:
                        response = client.get("http://127.0.0.1:8001/health", timeout=2)
                        if response.status_code == 200:
                            if response.json().get("candidate_sha256") != fingerprint(variant)["candidate_sha256"]:
                                raise ValueError("preview_fingerprint_mismatch")
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(.5)
                else:
                    raise ValueError("preview_startup_timeout")
                receipt = {"pid": process.pid, "variant": variant, "at": timestamp(), "parent": "reliability_v2_matrix"}
                immutable_write(ROOT / DATA / "process_history" / (log_id + ".json"), canonical(receipt))
                result = evaluate(variant, mode, split, passes)
                print(json.dumps({"completed": variant, "summary": result}), flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--mode", choices=("http", "controlled"), default="http")
    parser.add_argument("--split", choices=("development", "historical", "heldout_regression"), default="development")
    parser.add_argument("--passes", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    run(args.variants, args.mode, args.split, args.passes)
