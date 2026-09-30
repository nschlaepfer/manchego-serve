# manchego-serve 0.2.0: Manchego v3, with its temperature map and the CUDA fast path by default

**Status.** `0.2.0` on the local branch `fast-path`. It is not tagged, pushed, published or built into an image. The
published version is 0.1.1; 0.1.2 (the v2.1 temperature map) exists only on local `main`, and 0.2.0 builds on it.
The v3 Hub revisions are filled in: Manchego `f82e029d`, MLX-8bit `4ebdc0dd`, MLX-4bit `0c17e076` (tag `v3`, 2026-09-30); v3 is the default.

**What it serves.** Manchego v3, the new default model, and Manchego v2.1, each under its own published policy. The
model folder's `manchego_config.json` selects the prompt contract and the default fast path; the weights' hash
selects the packaged temperature map.

| | Manchego v3 | Manchego v2.1 |
|---|---|---|
| contract (`manchego_config.json`) | `semif` | `auto` (no field) |
| default temperature map (by `weights_sha256`) | `temperature_map_v3.json`: choice 1.5, noul 0.2, score 1.0 | `temperature_map_v2.1.json`: choice 1.791, noul 1.73, score 1.0 (unchanged) |
| default fast path (`serving`, CUDA only) | CUDA graphs + the lean host path | none |

**Unchanged for v2.1.** A folder without `contract` and `serving` fields (the published v2.1 folder) gets 0.1.2's
prompts, token ids, temperature map, answers, responses and `/healthz`, byte for byte, apart from the version string
(`"server": "manchego-serve 0.2.0"`). Two edge cases change, both towards the published policy:
with `--no-hash` and a bare directory the v2.1 map is not applied, and the reference linear-attention kernels run even
when flash-linear-attention is installed. The wire contract, the limits, the readout (one pass, the client's option
order, the confidence definition) and the refusals are unchanged for both models.

**Why these choices.** v3 is meant to score well on JevBench v1.5, where
- a noul answer counts as an abstention (scored wrong) unless P(yes) ≤ 0.20 or ≥ 0.80;
- speed is the mean of the p50 and p95 latency of serial single-decision requests;
- cost is proportional to input tokens.

## 1. Contract "semif", chosen per model

- **Where the contract comes from.** The model folder's `manchego_config.json` selects the prompt policy through a
  `contract` field (`model_config.py`). `--contract model|auto|semif` or `$MANCHEGO_CONTRACT` overrides it; the default
  is `model`, which reads the file.
  - No file, or no `contract` field, means `auto`: v2.1 keeps 0.1.2's prompts, token for token (`golden_prompts.json`).
  - An unreadable file or an unknown contract stops the server at start-up.
  - Optional `served_name` and `release_date` name the model in responses and in `/v1/models`. The default stays
    `manchego-2.1`.
- **What `semif` renders.**
  - A question with 2 to 16 options and a nonempty state gets SemIf's `direct-options-v1` prompt: a system message and
    one JSON user message `{evidence, criterion, options}`, letters A–P, noul as true then false.
  - Every other question gets the state-first prompt of `contract_v2.py`, never the short prompt: 17 to 255 options,
    or a state of `""`, `{}`, `[]` or `null`. This is the overflow rule of the research repository's development reads.
  - `contract_by_question` reports `semif` or `state_first` for each question.
- **Exactly the training code's rendering.**
  - `manchego_serve/contract_semif.py`'s code is the training code's renderer, unchanged. Its documentation was
    rewritten for publication (the file also ships next to the v3 weights): it no longer calls itself a draft that is
    "not the served contract", names no research-repository path, and credits SemIf. `tests/test_semif.py` checks the
    code, not the bytes: the module's source without docstrings, comments and blank lines must have the SHA-256
    recorded from the training renderer (`RENDERER_CODE_SHA256`). An optional test (`MANCHEGO_TEST_SEMIF_RENDERER=<the
    training file>`) re-derives that constant; it passes against the research repository's file (sha256 `86740b8d…`,
    commit `0da1e038`).
  - `tests/fixtures/golden_semif.json` (37 invented questions: 29 SemIf prompts and 8 v2 overflows, recorded from the
    training code alone) is unchanged: every prompt byte for byte without a tokenizer, every token id and option-code
    id with one (checked with the tokenizers of v2.1 and of the v3 MLX 8-bit build).
- **NOTICE** credits "SemIf (formerly OpenJev)", <https://github.com/TheoLeeCJ/SemIf>, with its MIT licence text as the
  local SemIf clone states it (`Copyright (c) 2026 TheoLeeCJ`). It no longer says `SemIf-OpenJev`.

## 2. Temperature maps

- **Schema 2** (`"schema": "manchego-temperature-map/2"`):
  - per-type temperatures in [0.05, 20], so values below 1 (sharpening) are allowed;
  - required `model_sha256` (the served weights' `weights_sha256`), and required `fitted_on` and `rule` texts;
  - an optional `contract`.
  - Passed as a file, a schema-2 map stops the server when the loaded weights' hash differs, when the weights were not
    hashed (`--no-hash`), or when `contract` differs from the served one. It is never applied "as asked".
- **The packaged v3 map (new).** `manchego_serve/temperature_map_v3.json` is the TMAP-V15 fit, byte for byte (sha256
  `56dae80c…`, pinned in `temperature.py`; the same bytes as the v3 release package's `temperature_map.json`):
  choice 1.5, noul 0.2, score 1.0, fitted under contract `semif` on 28,182 development rows v3 never trained on, by a
  rule registered before the fit (per type, the temperature on a grid 0.2 to 2.0 that maximises an estimate of
  JevBench v1.5's composite; no benchmark item).
  - `--temperature-map default` applies it to the v3 builds in `temperature.V3_BUILDS`, by `weights_sha256`: the bf16
    weights the file names (`2ee83843…`), and, bound by this package, the MLX 8-bit (`358b025b…`) and MLX 4-bit
    (`e1bc5538…`) builds, whose binding reads: "fitted on the bf16 path's development records; applied to the MLX
    builds as-is; chosen options are never changed by a temperature".
  - Only under contract `semif`, and never to unhashed weights. Every other model gets T = 1.0 unless a map is passed.
  - The file passed explicitly binds to the bf16 hash only, as any schema-2 file does.
  - `load("default")` checks the SHA-256 of both packaged files before the weights are read.
- **Why noul T = 0.2.** Under JevBench v1.5's band, a noul answer abstains when its yes/no logit margin is below
  T · ln 4: 1.39 at T = 1, 0.28 at T = 0.2. Sharpening never moves an answer across 0.5, so it turns would-be
  abstentions into Yes or No answers and changes no Yes into a No. Under the map P(yes) is not a calibrated
  probability; `--temperature-map off` gives T = 1. A test checks the arithmetic.
- **The v2.1 map** is unchanged, applies only to the v2.1 builds it lists under contract `auto`, and still carries the
  README warning: its noul T = 1.73 softens, so for v2.1 on JevBench v1.5 use `--temperature-map off`.

## 3. The fast path, and each model's default

`manchego_serve/backends/fast_path.py`: `--gdn-kernels reference|fast`, `--cuda-graphs` / `--graph-buckets`,
`--fast-host`, `--fast-path`, and (0.2.0) `--no-cuda-graphs`, `--no-fast-host`, `--no-fast-path`.
[FAST_PATH.md](FAST_PATH.md) has the details, the measurements and the validation commands.

- **(a) Kernels.** `fast` runs flash-linear-attention's gated delta rule and causal-conv1d where importable, reported
  per function. `reference`, the default, forces Transformers' PyTorch functions even when those packages are
  installed. The kernels are never a model default.
- **(b) CUDA graphs.** One-prompt forwards are replayed at right-padded fixed lengths (256 / 512 / 1024 / 2048), with no
  mask, no KV cache, and the read at the prompt's last position. Longer prompts and batches run eagerly.
- **(c) Lean host path.** Pinned non-blocking copies, no all-ones mask, cached candidate rows, one device-to-host copy
  per question. Bit for bit the reference.
- **Per-model defaults (new).** `manchego_config.json` may declare `"serving": {"cuda_graphs": true, "fast_host": true}`
  (optional `"graph_buckets"`). `settle` decides each setting: the command line, then the environment, then the
  model's defaults, then off. The defaults apply only to the torch backend on a CUDA device; elsewhere they are not
  applied and `/healthz` says so (`fast_path_settings.note`). Any other key in `serving` stops the server.
  `$MANCHEGO_CUDA_GRAPHS` and `$MANCHEGO_FAST_HOST` now take 1 or 0 (unset: the model decides; anything else is
  refused). `/healthz` reports `fast_path_settings` (what is in force and where each setting came from) and
  `model_config.serving`, only when the folder declares `serving` or a piece is on. A start-up failure of a fast path
  taken from the model folder names the flags that turn it off.
- **Validated on an RTX 5090 (Windows, 2026-09-30,** the v3 release candidate's bf16 weights, contract semif, the
  83 golden questions and 450 serial requests):
  - `fast_host`: max |Δp| 0, identical.
  - CUDA graphs: max |Δp| **0.0385** at the default buckets (0.0305 with 15 buckets, 0.0626 with 29), no chosen option
    changed.
  - fla kernels: max |Δp| 0.0562 and one changed answer (at a reference logit gap of 0.043; measured under contract
    auto): rejected.
  - Latency, graphs + fast host against the reference: p50 124.45 → 25.59 ms, p95 232.02 → 180.34 ms, mean of the two
    178.23 → 102.96 ms; prompts up to 256 tokens p50 114.93 → 24.75 ms, 257 to 512 tokens 141.54 → 43.56 ms.
- **The decision (the account holder's; disclosed, not a passed test).** The criterion set beforehand, max |Δp| ≤ 0.03,
  was exceeded by the graphs with no answer changed. The account holder accepted CUDA graphs + fast host as v3's
  default, disclosed next to the known cross-runtime differences (0.039 between two CPU platforms at bf16; up to 0.070
  between CUDA bf16 and the MLX 8-bit build on 255 options). The kernels stay off. v3's `manchego_config.json` declares
  the default; v2.1 declares none.

## 4. Manchego v3's weights, version names, the download and the image

- **Pins.** `weights.py` pins v3 in its three repositories: every weight-file and support-file SHA-256, computed from
  the release builds (`weights_sha256`: bf16 `2ee838433bfe278a226dc644667ad4a99ece82cc47325c7645a7dae723c1863b`, MLX
  8-bit `358b025b04001e50a065f8c87929175211264bd6182af74b67bd6caa2f639657`, MLX 4-bit
  `e1bc5538b8dced2a857b4980dba045c2ca01db1c369aa416fc19f0f5c593e782`; the support files are v2.1's, byte for byte).
  The revisions are placeholders until the Hub upload. A pin counts only once its revision is a full commit sha: until
  then v3 bytes match no pin (`weights_verified: false`, while the contract, the map and the fast path still follow
  the folder and the hash), and every default stays v2.1.
- **Version names.** `v3` and `v2.1` may stand for a revision in `manchego-serve-download --revision`,
  `manchego-serve --revision` and `$MANCHEGO_REVISION`. A name means the commit this package pins, never the Hub's tag.
  For a folder, the name is read for the backend's default repository.
- **`manchego-serve-download`** fetches the newest pinned version by default (v3 once `V3_REVISION` is filled in; it
  says so while it is not), keeps v2.1 available by name (`--revision v2.1`), and verifies the bytes of any pinned
  revision it fetches. `--revision v3` is refused while v3's revision is a placeholder.
- **Dockerfile.** `--build-arg MANCHEGO_VERSION=v3|v2.1` selects the model; `MODEL_REVISION` (a full sha) overrides the
  pinned commit. The default stays `v2.1` until `V3_REVISION` is filled in, then becomes `v3`: `tests/test_weights.py`
  fails until the Dockerfile's default follows the pin. The image declares the same revision at run time
  (`MANCHEGO_REVISION=${MODEL_REVISION:-${MANCHEGO_VERSION}}`), so a v2.1 image declares exactly the commit 0.1.x's did.
  The Dockerfile was not built for this release (no network here).

## Placeholders left

| placeholder | where | filled with |
|---|---|---|
| `REVISION_V3` | `manchego_serve/weights.py` (`V3_REVISION`); README (pinned weights table) | the full sha of the v3 commit on `oraculumai/Manchego` |
| `MLX8_REVISION_V3` | `manchego_serve/weights.py` (`V3_MLX8_REVISION`); README | the full sha of the v3 commit on `oraculumai/Manchego-MLX-8bit` |
| `MLX4_REVISION_V3` | `manchego_serve/weights.py` (`V3_MLX4_REVISION`); README | the full sha of the v3 commit on `oraculumai/Manchego-MLX-4bit` |

With them: change the Dockerfile's `ARG MANCHEGO_VERSION` default to `v3` (the test above requires it), replace the
README's "Until the v3 upload" note and the placeholder cells, update `test_v3_revisions_are_placeholders_until_the_upload`
(it asserts the placeholders), and run the suite.

## What could not be verified here

- 0.2.0 as a whole on CUDA: the RTX 5090 run used 0.2.0.dev0 (`be64e5b`), whose fast-path code 0.2.0 keeps. The new
  start-up choices (the folder's serving defaults, the packaged v3 map by hash, version names) were tested without CUDA:
  unit tests, a tiny random Qwen3.5 on CPU with CUDA stood in for, and the real v3 MLX 8-bit build on the Mac (served
  under contract semif with the v3 map bound by its hash, the serving defaults reported as not applied on MLX, a
  repeat-identical warm-up).
- A container image of 0.2.0 (building one downloads the weights).
- GPU memory with CUDA graphs, and the latency rows for prompts over 512 tokens: they are in the RTX 5090 run's raw
  files, not in this repository.

## Notes on reproducing the research renderer

- The server reproduces `semif_contract.render` as the development reads use it: every golden prompt and id, the slot
  order, and the v2 overflow. Nothing in the renderer was found that the server cannot reproduce.
- Three places differ from other research code paths:
  - **The lean trainer has no empty-state rule.** It overflows by option count only, so an empty-state row cannot be
    trained under SemIf. The server follows the reads' rule (empty state → contract v2), as specified.
  - **The research server's own `serve_systemone.render_semif` is a different, older renderer.** It keeps the client's
    noul order and concatenates noul descriptions without `_text`. The server follows `semif_contract.py`, as the
    trainer and the reads do.
  - **The reads score in padded batches** (64 rows, 8,192 padded tokens), and the trainer runs the adapter unmerged.
    The server scores one prompt per pass on merged weights. The v3 candidate's own `merge_record.json` reports a merge
    check of max |Δlogit| 0.125 (12 rows).

## Tests

The suite runs with `pytest` from the repository root (pyproject: `pythonpath = [".", "tests"]`), in two local
environments: one with fastapi, httpx, transformers 5.17.0 and mlx-lm but no torch, and one with torch 2.10.0 and
transformers 5.17.0 but no fastapi (both Python 3.12.11). The tokenizer is v2.1's (`MANCHEGO_TEST_TOKENIZER`).

| run | 0.1.2 (fcf464c) | 0.2.0.dev0 (be64e5b) | 0.2.0 |
|---|---:|---:|---:|
| fastapi env, no tokenizer | 108 passed, 51 skipped | 224 passed, 103 skipped | 265 passed, 110 skipped |
| fastapi env, tokenizer | 155 passed, 4 skipped | 309 passed, 18 skipped | 350 passed, 25 skipped |
| torch env, no tokenizer | 104 passed, 55 skipped | 229 passed, 98 skipped | 269 passed, 106 skipped |
| torch env, tokenizer | 151 passed, 8 skipped | 318 passed, 9 skipped | 364 passed, 11 skipped |

- **Why tests skip.** The other environment's dependency (torch or fastapi), the numeric tests that need weights, and
  the optional training-renderer test (`MANCHEGO_TEST_SEMIF_RENDERER`).
- **Changed tests.** One existing test was replaced, as the release asked: the byte pin of `contract_semif.py`
  (`test_vendored_renderer_is_the_training_file`) became a check of its code (`test_renderer_code_is_the_training_code`).
  Every other existing test is unchanged.
- **New tests** cover the serving defaults (settle, the command line, `/healthz`, and `server.build` on a tiny model
  with CUDA stood in for), the packaged v3 map (its bytes, bindings, contract, `--no-hash`, a response, the noul band),
  the v3 pins and placeholders, version names, the download, the Dockerfile's default, the renderer's documentation
  and NOTICE's SemIf credit.
- **Real weights** (not in the counts above):
  - `MANCHEGO_TEST_MLX_MODEL=` the pinned v2.1 MLX 8-bit folder: `test_numeric.py` reproduces the v0.1.0 golden answers
    byte for byte in both executions (max difference 0; 3 passed, the torch informational test skipped).
  - `test_semif.py` and `test_parity.py` with the v3 MLX 8-bit tokenizer and the training renderer: 211 passed.

## Version

`pyproject.toml` and `manchego_serve.__version__` are `0.2.0`. Responses say `"server": "manchego-serve 0.2.0"`.


**Filled in 2026-09-30:** `V3_REVISION = f82e029d0ad4d1bdd1ca12f5f7b548b84fabc9a9`, `V3_MLX8_REVISION = 4ebdc0dd50481cfa9e40f83f05571ee59e53041d`, `V3_MLX4_REVISION = 0c17e076e8e358e41a9f8332dca062e4942ace0b` (the verified upload commits); the Dockerfile default is `v3`; the README note and table cells are updated.
