from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


OWNER = "pabloitu"
REPO  = "fem2geo"
_API  = "https://api.github.com"


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "download-tutorials",
        help="Download the tutorials bundle from the latest release into ./tutorials.",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    dest = Path.cwd() / "tutorials"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    release = _get_json(f"{_API}/repos/{OWNER}/{REPO}/releases/latest")
    tag     = release.get("tag_name", "<unknown>")
    assets  = release.get("assets", [])
    name    = f"fem2geo-tutorials-{tag}.zip"

    url = next((a["browser_download_url"] for a in assets if a.get("name") == name), None)
    if not url:
        available = [a.get("name") for a in assets]
        print(f"Asset '{name}' not found in latest release ({tag}).", file=sys.stderr)
        print(f"Available: {available}", file=sys.stderr)
        return 2

    print(f"Release: {tag}")
    print(f"Downloading: {name}")
    data = _get_bytes(url)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)

    print(f"Tutorials extracted to: {dest}")
    return 0


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()