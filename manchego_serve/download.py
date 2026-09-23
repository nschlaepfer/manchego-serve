# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""manchego-serve-download: fetch a pinned Manchego revision ONCE (setup / image build time) and verify its bytes.

This is the only part of the package that uses the network. The server itself runs with HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    from .weights import PINS, identify
    ap = argparse.ArgumentParser(prog="manchego-serve-download", description=__doc__)
    ap.add_argument("--repo", default="oraculumai/Manchego", help="hub id (default: oraculumai/Manchego)")
    ap.add_argument("--revision", default=None, help="full commit sha (default: the pinned v2.1 revision of --repo)")
    ap.add_argument("--out", default=None, help="download into this directory instead of the Hugging Face cache")
    args = ap.parse_args(argv)
    pin = PINS.get(args.repo)
    revision = args.revision or (pin and pin["revision"])
    if not revision:
        ap.error(f"--revision is required for {args.repo} (no pin known)")
    if len(revision) != 40:
        print(f"warning: {revision!r} is not a full commit sha; tags and branches can move", file=sys.stderr)
    from huggingface_hub import snapshot_download
    path = snapshot_download(args.repo, revision=revision, local_dir=args.out)
    ident = identify(path)
    print(json.dumps({"repo": args.repo, "revision": revision, "path": path, **ident}, indent=1))
    if pin and revision == pin["revision"] and ident["weights_match_pin"] is None:
        print(f"error: the downloaded weight files do not match the pinned SHA-256 of {args.repo}@{revision}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
