"""Generate the small static frontend deployment with an exact Render origin."""
import argparse
import json
from pathlib import Path
import shutil

from ipindia_corpus.cloud_config import _origin
from ipindia_corpus.http_security import CSP
from ipindia_corpus.sync_runner import ROOT

ASSETS = {".css", ".js", ".svg", ".png", ".jpeg", ".ttf", ".txt"}
APIS = "live|health|search|sources|capabilities|answer"


def prepare(output, render_origin, root=ROOT):
    origin = _origin(render_origin)
    if not origin.removeprefix("https://").endswith(".onrender.com"):
        raise ValueError("render_origin_required")
    output = Path(output).resolve()
    root = Path(root).resolve()
    if output.exists() or output == root or root.is_relative_to(output):
        raise ValueError("new_separate_staging_directory_required")
    output.mkdir(parents=True)
    web = root / "web"
    for name in ("index.html", "research.html"):
        shutil.copyfile(web / name, output / name)
    for asset in sorted(web.rglob("*")):
        if asset.is_file() and asset.suffix in ASSETS:
            destination = output / "assets" / asset.relative_to(web)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset, destination)

    headers = {
        "Content-Security-Policy": CSP,
        "Strict-Transport-Security": "max-age=31536000",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    }
    no_cache = {
        "Cache-Control": "no-store",
        "CDN-Cache-Control": "no-store",
        "Vercel-CDN-Cache-Control": "no-store",
    }
    configuration = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "framework": None,
        "routes": [
            {"src": "/(.*)", "headers": headers, "continue": True},
            {
                "src": f"/({APIS})",
                "dest": origin + "/$1",
                "headers": no_cache,
                "transforms": [{
                    "type": "request.headers", "op": "set", "target": {"key": "x-origin-secret"},
                    "args": "$ORIGIN_SECRET", "env": ["ORIGIN_SECRET"],
                }],
            },
            {"src": "/", "dest": "/index.html"},
            {"src": "/research/?", "dest": "/research.html"},
            {"handle": "filesystem"},
        ],
    }
    (output / "vercel.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render-origin", required=True)
    args = parser.parse_args()
    print(prepare(args.output, args.render_origin))
