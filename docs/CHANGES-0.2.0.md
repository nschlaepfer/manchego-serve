# manchego-serve 0.2.0.dev0: ready for the next model (development version, NOT a release)

**Status.** `0.2.0.dev0` on the local branch `fast-path`. It is not tagged, published, pushed or built into an image.
The published version is 0.1.1; 0.1.2 (the temperature map) exists only on local `main`, and this branch builds on it.

**Purpose.** Serve the next model, Manchego v3 (Qwen3.5-4B + LoRA trained under the SemIf prompt contract), so that it
scores well on JevBench v1.5. Three facts about that benchmark shaped the changes:
- a noul answer counts as an abstention (scored wrong) unless P(yes) ≤ 0.20 or ≥ 0.80;
- speed is the mean of the p50 and p95 latency of serial single-decision requests;
- cost is proportional to input tokens.

The wire contract, the limits, the readout (one pass, the client's option order, the confidence definition) and the
refusals are unchanged.

## 1. Contract "semif", chosen per model

- **Where the contract comes from.** The model folder's `manchego_config.json` now selects the prompt policy through a
  `contract` field (`model_config.py`). `--contract model|auto|semif` or `$MANCHEGO_CONTRACT` overrides it; the default
  is `model`, which reads the file.
  - No file, or no `contract` field, means `auto`. The published v2.1 folder has no such field, so v2.1 keeps 0.1.2's
    prompts, token for token (`golden_prompts.json` still passes).
  - An unreadable file or an unknown contract stops the server at start-up.
  - Optional `served_name` and `release_date` name the model in responses and in `/v1/models`. The default stays
    `manchego-2.1`.
- **What `semif` renders.**
  - A question with 2 to 16 options and a nonempty state gets SemIf's `direct-options-v1` prompt. That is a system
    message and one JSON user message `{evidence, criterion, options}`, letters A–P, noul as true then false.
  - Every other question gets the state-first prompt of `contract_v2.py`, never the short prompt: 17 to 255 options,
    or a state of `""`, `{}`, `[]` or `null`. This is the v2 overflow rule of the research repository's development
    reads (`scripts/lean_reads_score.py`, `encode(..., "semif")`).
  - The limits (2 to 255 options) and the refusals are unchanged. `contract_by_question` reports `semif` or
    `state_first` for each question.
- **Exactly the training code's rendering.**
  - `manchego_serve/contract_semif.py` is the research repository's `qwen_decisions/semif_contract.py` (commit
    `0da1e038`), byte for byte below a three-line release note. A test pins its sha256 and git blob id.
  - `tests/fixtures/golden_semif.json` was recorded by `tests/record_semif_golden.py` from the training code alone
    (tokenizer only, no weights). It covers 37 invented questions: 29 SemIf prompts and 8 v2 overflows. Three training
    paths agree on every case: the renderer, the lean trainer's `LeanEncoder.encode`, and the development reads'
    `encode()`.
  - `tests/test_semif.py` checks every prompt byte for byte without a tokenizer, and every token id and option-code id
    with one. The 23 cases recorded earlier on this branch are unchanged.
- **NOTICE.** It credits SemIf-OpenJev (MIT) for the prompt format and system message text.

## 2. Temperature map schema 2; T = 1.0 for any model without a bound map

- **Schema 2** (`"schema": "manchego-temperature-map/2"`):
  - per-type temperatures in [0.05, 20], so values below 1 (sharpening) are allowed;
  - required `model_sha256` (the served weights' `weights_sha256`), and required `fitted_on` and `rule` texts;
  - an optional `contract`.
  - The server refuses to start when the loaded weights' hash differs, when the weights were not hashed (`--no-hash`),
    or when `contract` differs from the served one. Such a map is never applied "as asked".
- **The default (`resolve`).** A map applies only to weights it is bound to by hash.
  - The packaged v2.1 map (unchanged; sha256 still pinned) applies only to the v2.1 builds it lists, and only under
    contract `auto`. Every other model, v3 included, gets T = 1.0.
  - With `--no-hash`, a bare directory with nothing declared now gets T = 1.0 (0.1.2 kept the map, reported as not
    checked). A declared, listed v2.1 build keeps 0.1.2's behaviour.
  - Schema-1 files passed explicitly behave as in 0.1.2.
- **No map ships for v3.** Its values come later, as a schema-2 file fitted on the served (merged) weights' logits.
- **README warning.** The v2.1 map's noul T = 1.73 moves noul answers towards 0.5. Under JevBench v1.5's
  0.20 / 0.80 band, every noul answer with a yes/no logit margin between 1.39 and 2.40 becomes an abstention, scored
  wrong. For v2.1 on that benchmark, use `--temperature-map off`. A test checks the arithmetic.

## 3. The opt-in fast path (CUDA), every piece OFF by default

`manchego_serve/backends/fast_path.py`, with flags `--gdn-kernels reference|fast`, `--cuda-graphs` / `--graph-buckets`,
`--fast-host` and `--fast-path` (all three). [FAST_PATH.md](FAST_PATH.md) has the details and the validation commands.

- **(a) Kernels.**
  - `fast` runs flash-linear-attention's gated delta rule and causal-conv1d in Qwen3.5's linear-attention layers,
    each where importable and reported per function.
  - `reference`, the default, now forces Transformers' PyTorch functions even when those packages are installed. 0.1.x
    used them silently whenever Transformers could import them.
- **(b) CUDA graphs.** One-prompt forwards are replayed at right-padded fixed lengths (256 / 512 / 1024 / 2048), with no
  mask, no KV cache, and the read at the prompt's last position. Longer prompts and batches run eagerly.
- **(c) Lean host path.**
  - It uses pinned non-blocking copies, no all-ones mask (Transformers checks one on the host twice per forward, then
    drops it), cached candidate rows, and one device-to-host copy per question. It is bit for bit the reference.
  - Profiling found no per-request template or tokenizer rebuild to remove. Transformers caches the compiled chat
    template: 0.05 ms per question, and tokenisation 0.2 to 0.3 ms, against hundreds of milliseconds for a forward on
    MPS.
- **The mask.** The mask is passed as Transformers' precomputed `{layer type: None}` mapping. That is what Transformers
  builds from an all-ones mask. It also stops `is_tracing()`, which is true during stream capture, from switching SDPA
  to an explicit mask inside a graph.
- **Reporting.** With everything off, `logits_batch` is the 0.1.x code. `/healthz` reports the kernels under
  `runtime.gdn_kernels`; responses carry `manchego.fast_path` only when a piece is on.
- **`tools/fast_path_check.py`.** `agree` loads the model once and compares each configuration with the reference on
  the 83 golden questions: `fast_host` must be exactly equal; the others must stay within max |Δp| ≤ 0.03 with no
  change of choice. `latency` sends 450 serial single-decision requests and reports p50, p95 and their mean.

**Profiling on the Mac** (the local v3 release candidate, bf16 weights, reference kernels; informational):
- MPS forward 353 / 724 / 1412 / 2838 / 6217 ms at 128 / 256 / 512 / 1024 / 2048 tokens. The chunked gated delta rule
  takes 61% of it; host-to-device copies about 1 ms.
- CPU float32 spends 88% of its forward in the depthwise `F.conv1d` (324 ms per call at 256 tokens, against 1.1 ms for
  an equivalent shift-and-add). This is a CPU-only finding, not changed here.
- MPS preview of `agree`, run on the 83 golden questions under SemIf:
  - `--fast-host`: max |Δp| exactly 0.
  - Graph padding through an eager stand-in: max |Δp| 0.0171, with no change of choice.
  - The first run of this preview found a real bug, now fixed and tested: a non-blocking copy from pageable memory,
    which read freed memory on MPS and gave max |Δp| 0.96.

## What could not be verified here (no CUDA)

- Whether the forward captures as a CUDA graph at all. A host sync inside a kernel package would fail the capture;
  the server then refuses to start and names the error. So is the speed of the graphs.
- fla and causal-conv1d on the real model: the arithmetic change and the speed.
- The whole-server latency and GPU memory with the fast path.

These are the steps of FAST_PATH.md. Until they are recorded, the flags stay opt-in, and a configuration that fails
`agree` is not served.

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
    check of max |Δlogit| 0.125 (12 rows). A v3 temperature map should therefore be fitted on the served weights'
    sequential logits and bound to their hash, which is what schema 2 enforces.

## Tests

The suite runs with `pytest` from the repository root (pyproject: `pythonpath = [".", "tests"]`). There are two local
environments: one with fastapi, httpx, transformers 5.17.0 and mlx-lm but no torch, and one with torch 2.10.0 and
transformers 5.17.0 but no fastapi. The tokenizer is v2.1's (`MANCHEGO_TEST_TOKENIZER`).

| run | 0.1.2 (fcf464c) | 0.2.0.dev0 |
|---|---:|---:|
| fastapi env, no tokenizer | 108 passed, 51 skipped | 224 passed, 103 skipped |
| fastapi env, tokenizer | 155 passed, 4 skipped | 309 passed, 18 skipped |
| torch env, no tokenizer | 104 passed, 55 skipped | 229 passed, 98 skipped |
| torch env, tokenizer | 151 passed, 8 skipped | 318 passed, 9 skipped |

- **Why tests skip.** The remaining skips are the other environment's dependency (torch or fastapi), and the numeric
  tests that need weights.
- **Unchanged tests.** No existing test was changed.
- **New test files.** `test_semif.py`, `test_temperature_v2.py`, `test_fast_path.py` and `test_build.py`.
- **Real weights** (not in the counts above):
  - `MANCHEGO_TEST_MLX_MODEL=` the pinned v2.1 MLX 8-bit folder: `test_numeric.py` reproduces the v0.1.0 golden answers
    byte for byte in both executions (max difference 0; 3 passed, the torch informational test skipped).
  - The SemIf and prompt-parity id tests also pass with the MLX 8-bit tokenizers of v2.1 and of the v3 candidate.
  - `tests/test_build.py` (4 tests, torch and tokenizer) builds a server Decider from a tiny model folder.

## Version

`pyproject.toml` and `manchego_serve.__version__` are `0.2.0.dev0` (a PEP 440 development release). Responses say
`"server": "manchego-serve 0.2.0.dev0"`.
