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
REPO = "fem2geo"
GITHUB_API = "https://api.github.com"


def download_tutorials(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "download-tutorials",
        help="Download and extract the tutorials bundle into ./tutorials.",
    )
    p.set_defaults(func=_download_tutorials)


def _download_tutorials(args: argparse.Namespace) -> int:
    dest = Path.cwd() / "tutorials"

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    release = _get_json(f"{GITHUB_API}/repos/{OWNER}/{REPO}/releases/latest")
    tag = release.get("tag_name", "<unknown>")
    assets = release.get("assets", [])

    expected = f"fem2geo-tutorials-{tag}.zip"
    url = None
    for a in assets:
        if a.get("name") == expected:
            url = a.get("browser_download_url")
            break

    if not url:
        available = [a.get("name") for a in assets]
        print(
            f"Could not find '{expected}' in latest release assets.\n"
            f"Available assets: {available}",
            file=sys.stderr,
        )
        return 2

    print(f"Latest release: {tag}")
    print(f"Downloading: {expected}")

    zip_bytes = _get_bytes(url)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest)

    print(f"Extracted tutorials to: {dest}")
    return 0


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()