# src/fem2geo/cmd/main.py
from __future__ import annotations

import argparse
import sys

from fem2geo.cmd.tutorials import download_tutorials


def fem2geo(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fem2geo",
        description="fem2geo command line interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_tutorials(subparsers)

    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(fem2geo(sys.argv[1:]))