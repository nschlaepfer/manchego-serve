"""Record the SemIf contract's expected renderings from the TRAINING code (not a test; run once, needs the training repo).

    /path/to/training/.venv/bin/python tests/record_semif_golden.py \
        --training-repo /path/to/qwen-decisions-torch --tokenizer /path/to/Qwen3.5-4B-or-Manchego-folder

Needs the training repo and a tokenizer folder; no model weights are read. For every invented request in
fixtures/semif_requests.json this renders the question the way the SemIf-trained models are trained and read, through
three training paths that must agree:

  1. qwen_decisions.semif_contract.render / prompt_text on serve_systemone.options_of(question) (the renderer and the
     System One mapping its docstring names);
  2. qwen_decisions.lean_arm.LeanEncoder.encode(row, "semif", identity order) (what the trainer tokenizes);
  3. scripts/lean_reads_score.py encode(enc, tok, row, "semif") (what the development reads score). A row SemIf cannot
     show (fewer than 2 or more than 16 options, an empty state, a noul row without true/false) is shown under contract
     v2 there (`shown` 'v2', lean_arm's overflow rule 'v2'); those rows are recorded with their contract v2 rendering
     (serve_systemone.RENDER["v2"], the codes of contract_v2.codebook) and the renderer's refusal message.

It writes fixtures/golden_semif.json: the renderer file's sha256 and git blob id, the training repo commit of each
training file used, the tokenizer's file hashes, and per request the contract it is shown under, the messages, codes,
keys, slot order, chat-templated prompt, token ids and option-code ids. Nothing from manchego_serve is imported here, so
the expected side is the training code alone. tests/test_semif.py compares the service with this file; the service
never imports the training repo.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def committed(repo: Path, path: Path) -> dict:
    """sha256, git blob and last commit of a training file, refusing one with uncommitted changes."""
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", str(path)], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{path} has uncommitted changes; record from committed training code")
    commit = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", str(path)], capture_output=True, text=True).stdout.strip()
    raw = path.read_bytes()
    return {"file": str(path.relative_to(repo)), "sha256": hashlib.sha256(raw).hexdigest(), "git_blob": git_blob(raw), "commit": commit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-repo", required=True)
    ap.add_argument("--tokenizer", required=True, help="a local folder with the Qwen3.5 tokenizer and chat template")
    ap.add_argument("--out", default=str(HERE / "fixtures" / "golden_semif.json"))
    a = ap.parse_args()
    repo = Path(a.training_repo).resolve()
    sys.path.insert(0, str(repo))
    from qwen_decisions import lean_arm as LA          # noqa: E402  (recorder only)
    from qwen_decisions import semif_contract as SC    # noqa: E402
    from qwen_decisions import serve_systemone as SO   # noqa: E402
    from qwen_decisions import v2_train as V2T         # noqa: E402
    from transformers import AutoTokenizer             # noqa: E402
    spec = importlib.util.spec_from_file_location("lean_reads_score", repo / "scripts" / "lean_reads_score.py")
    LRS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LRS)

    files = {"renderer": committed(repo, Path(SC.__file__)), "lean_arm": committed(repo, Path(LA.__file__)),
             "lean_reads_score": committed(repo, repo / "scripts" / "lean_reads_score.py"),
             "serve_systemone": committed(repo, Path(SO.__file__)), "contract_v2": committed(repo, repo / "qwen_decisions" / "contract_v2.py"),
             "v2_train": committed(repo, Path(V2T.__file__))}
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tdir = Path(a.tokenizer)
    lean = LA.LeanEncoder(tok, LA.LeanConfig(train_path="", dev_path="", out_dir=""))
    v2enc = V2T.Encoder(tok)

    reqs = json.loads((HERE / "fixtures" / "semif_requests.json").read_text(encoding="utf-8"))["requests"]
    cases = {}
    for r in reqs:
        cid, state, q = r["id"], r["state"], r["question"]
        opts = SO.options_of(q)
        row = {"id": cid, "kind": q["type"], "state": state, "instructions": q["instructions"], "options": opts,
               "target": {str(opts[0][0]): 1.0}}
        read = LRS.encode(v2enc, tok, row, "semif")                          # path 3: the reads' encoding, overflow included
        if read["shown"] == "semif":
            out = SC.render(state, q["instructions"], opts, q["type"])      # path 1
            text = SC.prompt_text(tok, state, q["instructions"], opts, q["type"])
            ids = tok.encode(text, add_special_tokens=False)
            code_ids = [lean.code_id(L) for L in out["letters"]]
            trained = lean.encode(row, "semif", list(range(len(opts))))      # path 2
            assert trained["ids"] == ids and trained["cands"] == code_ids, cid
            assert trained["order"] == [[str(v) for v, _ in opts].index(k) for k in out["keys"]], cid
            assert read["ids"] == ids and read["cands"] == code_ids and read["order"] == trained["order"], cid
            cases[cid] = {"shown": "semif", "messages": out["messages"], "codes": out["letters"], "keys": out["keys"],
                          "order": read["order"], "prompt_version": out["prompt_version"], "prompt": text, "ids": ids, "code_ids": code_ids}
            continue
        try:
            SC.render(state, q["instructions"], opts, q["type"])
            refusal = None
        except ValueError as e:
            refusal = str(e)
        assert read["shown"] == "v2", cid
        msgs = SO.RENDER["v2"](state, {"type": q["type"], "instructions": q["instructions"]}, [(str(v), d) for v, d in opts])
        codes = SO.codes_for("v2", len(opts))
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        assert read["ids"] == ids and read["cands"] == [v2enc.code_id(c) for c in codes], cid
        cases[cid] = {"shown": "v2", "semif_refusal": refusal, "messages": msgs, "codes": codes, "keys": [str(v) for v, _ in opts],
                      "order": read["order"], "prompt": text, "ids": ids, "code_ids": read["cands"]}
    golden = {
        "note": "Expected SemIf renderings (and the contract v2 renderings of the rows SemIf cannot show), recorded from the training "
                "code by tests/record_semif_golden.py. Invented requests only.",
        "renderer": {**files["renderer"], "prompt_version": SC.PROMPT_VERSION},
        "training_files": files,
        "training_paths": ["semif_contract.render + prompt_text on serve_systemone.options_of",
                           "lean_arm.LeanEncoder.encode(row, 'semif', identity order)",
                           "scripts/lean_reads_score.py encode(V2.Encoder, tok, row, 'semif'), including the v2 overflow"],
        "overflow_rule": "a row SemIf cannot show (fewer than 2 or more than 16 options, an empty state, a noul row without true/false) "
                         "is shown under contract v2 (lean_reads_score.semif_can_show)",
        "tokenizer": {n: (sha256(tdir / n) if (tdir / n).is_file() else None)
                      for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
        "cases": cases,
    }
    Path(a.out).write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shown = [c["shown"] for c in cases.values()]
    print(f"wrote {a.out}: {shown.count('semif')} SemIf renderings, {shown.count('v2')} shown under contract v2")


if __name__ == "__main__":
    main()
