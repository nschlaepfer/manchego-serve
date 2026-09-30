# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""manchego-serve-download: fetch a pinned Manchego revision ONCE (setup / image build time) and verify its bytes.

    manchego-serve-download                                   # the default version's pinned commit of oraculumai/Manchego
    manchego-serve-download --revision v2.1                   # Manchego v2.1 by name (this package's pinned commit)
    manchego-serve-download --repo oraculumai/Manchego-MLX-8bit --revision v3

The default version is the newest one whose revision this package pins: Manchego v3 once weights.V3_REVISION (and, for
the MLX repositories, V3_MLX8_REVISION / V3_MLX4_REVISION) holds the commit of the v3 upload, v2.1 until then. A version
name means this package's pinned commit, never the Hub's tag of that name.

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
    from . import weights as W
    ap = argparse.ArgumentParser(prog="manchego-serve-download", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="oraculumai/Manchego", help="hub id (default: oraculumai/Manchego)")
    ap.add_argument("--revision", default=None,
                    help=f"a full commit sha, or a Manchego version this package pins ({', '.join(W.VERSIONS)}) for its pinned "
                         "commit. Default: the newest version pinned for --repo (v3 once its revision is filled in, else v2.1)")
    ap.add_argument("--out", default=None, help="download into this directory instead of the Hugging Face cache")
    args = ap.parse_args(argv)
    known = any(args.repo in pins for pins in W.RELEASES.values())
    if args.revision is None and not known:
        ap.error(f"--revision is required for {args.repo} (no pin known)")
    try:
        revision = W.pinned_revision(args.repo, args.revision) if args.revision is None or args.revision in W.RELEASES else args.revision
    except W.RevisionError as e:
        ap.error(str(e))
    if args.revision is None:
        v3 = W.V3_PINS.get(args.repo)
        if v3 and not W.is_pinned(v3["revision"]):
            print(f"note: Manchego v3's revision of {args.repo} is not pinned in this build yet (placeholder "
                  f"{v3['revision']!r}); downloading {W.default_version(args.repo)}", file=sys.stderr)
    if not W.is_pinned(revision):
        print(f"warning: {revision!r} is not a full commit sha; tags and branches can move", file=sys.stderr)
    version, pin = W.pin_for(args.repo, revision)
    from huggingface_hub import snapshot_download
    path = snapshot_download(args.repo, revision=revision, local_dir=args.out)
    ident = W.identify(path)
    print(json.dumps({"repo": args.repo, "version": version, "revision": revision, "path": path, **ident}, indent=1))
    match = ident["weights_match_pin"] or {}
    if pin and (match.get("repo"), match.get("revision")) != (args.repo, revision):
        print(f"error: the downloaded weight files do not match the pinned SHA-256 of {args.repo}@{revision}", file=sys.stderr)
        return 1
    if pin and not ident["support_files_match_pin"]:
        print(f"error: the downloaded chat template / tokenizer / config files do not match the pinned SHA-256 of {args.repo}@{revision}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
