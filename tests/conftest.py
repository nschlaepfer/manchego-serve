import json
import os
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


REQUESTS = load("requests.json")["requests"]
BY_ID = {r["id"]: r for r in REQUESTS}


def local_or_cached(env: str, repo: str, revision: str):
    """A weights / tokenizer directory from `env`, else the pinned snapshot if it is already in the local HF cache."""
    p = os.environ.get(env)
    if p:
        return p
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo, revision=revision, local_files_only=True)
    except Exception:
        return None


@pytest.fixture(scope="session")
def tokenizer():
    from manchego_serve.weights import PINS
    path = local_or_cached("MANCHEGO_TEST_TOKENIZER", "oraculumai/Manchego", PINS["oraculumai/Manchego"]["revision"])
    if not path:
        pytest.skip("set MANCHEGO_TEST_TOKENIZER to a Manchego v2.1 folder (any of the three formats)")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


class TokOnly:
    """A backend with a tokenizer and no model: enough to plan (render + encode) a request."""
    name = "tokenizer-only"

    def __init__(self, tok):
        self.tok = tok
