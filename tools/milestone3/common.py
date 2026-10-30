"""Content-addressed local artifacts and protected historical inputs."""
import hashlib
import json
from pathlib import Path

from ipindia_corpus.llm_experiment import ROOT, canonical, timestamp
from ipindia_corpus.paths import immutable_write

DEST = Path("reports/milestone3")
DATA = Path("data/milestone3")
GENERATION = "20261014T204940358650Z"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(name, value, root=ROOT):
    raw = value.encode("utf-8") if isinstance(value, str) else canonical(value)
    checksum = sha(raw)
    immutable_write(Path(root) / DATA / "artifacts" / checksum, raw)
    path = Path(root) / DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(checksum + "\n", encoding="ascii")
    return checksum


def observations(root=ROOT):
    """Saved attempts only. A report hash binds every yielded observation."""
    for milestone in ("reliability", "reliability_v2"):
        for mode in ("controlled", "http"):
            for split in ("historical", "development", "heldout_regression"):
                suffix = "heldout" if milestone == "reliability" and split == "heldout_regression" else split
                prefix = "coverage-" if milestone == "reliability_v2" else ""
                path = Path(root) / "reports" / milestone / f"{prefix}{mode}-{suffix}-3passes.json"
                raw = path.read_bytes()
                expected = path.with_suffix(path.suffix + ".sha256").read_text().strip()
                if sha(raw) != expected:
                    raise ValueError("historical_report_hash_mismatch")
                for row in json.loads(raw)["rows"]:
                    yield {"milestone": milestone, "mode": mode, "split": split,
                           "report": path.relative_to(root).as_posix(), "report_sha256": sha(raw),
                           "row_sha256": sha(canonical(row)), **row}


def protected(root=ROOT):
    before = read(Path(root) / DEST / "preflight.json")
    changed = [name for name, expected in before["files"].items()
               if not (Path(root) / name).is_file() or sha((Path(root) / name).read_bytes()) != expected]
    return {"at": timestamp(), "snapshot_sha256": before["snapshot_sha256"],
            "checked_files": len(before["files"]), "changed": changed, "passed": not changed}
