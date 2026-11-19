"""Download the exact allowlisted E5 files at image build time."""
from pathlib import Path
import json
import sys

from huggingface_hub import snapshot_download

from ayurip_e5 import MODEL_ID, REVISION, verify_snapshot

root = Path(sys.argv[1]).resolve()
lock_path = Path(__file__).with_name("embedding-files.lock.json")
files = json.loads(lock_path.read_bytes())["files"]
snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=root, allow_patterns=list(files))
verify_snapshot(root, lock_path)
