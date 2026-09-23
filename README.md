# manchego-serve

A small, offline HTTP server for **[Manchego v2.1](https://huggingface.co/oraculumai/Manchego)**, a 4B decision model,
speaking TypeSafe AI's **System One wire contract**: `POST /v1/systemone`, `GET /v1/models`, `GET /healthz`.

It serves exactly the policy behind the model card's numbers: contract **auto** (the short prompt up to 26 options, the
state-first prompt for 27 to 255), **one pass** per question, **temperature 1.0**, the client's option order, confidence
**(K · max p − 1) / (K − 1)**, and every question scored as **its own sequence**. It never opens an outbound connection:
the weights are fetched once at setup (pinned by full commit sha, checked against the published SHA-256 of every weight
file and of the chat template, tokenizer and config files) and the server runs with `HF_HUB_OFFLINE=1`. No telemetry.

Runs on Linux + NVIDIA CUDA (PyTorch, bf16), on CPU (PyTorch, float32), and on Apple silicon (MLX, optional).

**Tested so far:** the MLX backend on Apple silicon, the PyTorch backend on CPU (macOS, and the CPU image on linux/arm64
with no network). The CUDA image has not yet been built or run on an NVIDIA GPU; its PyTorch code path is the one tested
on CPU, but bf16 on CUDA is unmeasured here.

## Quick start

### Docker (Linux, NVIDIA GPU)

```bash
docker build -t manchego-serve:2.1-cuda .          # downloads and verifies the weights (9.3 GB) into the image
docker run --rm --gpus all -p 127.0.0.1:8000:8000 manchego-serve:2.1-cuda
curl -s localhost:8000/healthz
```

The base is `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` (pinned by digest; linux/amd64; torch built for CUDA 12.8,
so an NVIDIA driver of the R570 series or newer is recommended, and required for RTX 50-series cards), plus the pinned
packages in `docker/requirements.txt`. The port opens only after the model is loaded and warmed up. The CUDA base image
is amd64-only: on an arm64 machine add `--platform linux/amd64` to build it.

CPU image (amd64 or arm64; float32 needs about 20 GB of RAM, `--dtype bfloat16` about 10 GB):

```bash
docker build -t manchego-serve:2.1-cpu \
  --build-arg BASE_IMAGE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
docker run --rm -p 127.0.0.1:8000:8000 manchego-serve:2.1-cpu                      # float32
docker run --rm -p 127.0.0.1:8000:8000 manchego-serve:2.1-cpu --dtype bfloat16     # about half the memory
```

Arguments after the image name are appended to `manchego-serve`. The image sets `MANCHEGO_HOST=0.0.0.0` and
`MANCHEGO_PORT=8000`, so extra arguments keep the container reachable.

Weights outside the image: build with `--build-arg DOWNLOAD_WEIGHTS=0` and mount a folder holding the pinned revision:
`docker run ... -v /path/to/Manchego:/models/manchego:ro manchego-serve:2.1-cuda`. The server hashes what it loads and
reports whether the bytes are the published ones (`weights_verified` for the weight files, `support_files_verified` for
the chat template, tokenizer and config files; in `/healthz` and in every response).

### pip

```bash
git clone <this repository> manchego-serve && cd manchego-serve
# Linux + CUDA: install the CUDA build of torch first
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install .
manchego-serve-download                  # once: oraculumai/Manchego @ 77403228..., verified (uses the network)
manchego-serve --backend torch --host 127.0.0.1 --port 8000     # offline from here on
```

Apple silicon: `pip install ".[mlx]"`, `manchego-serve-download --repo oraculumai/Manchego-MLX-8bit`, then
`manchego-serve --backend mlx`. `--model` also takes a local folder (`--model /path/to/Manchego --revision <sha>`).

| option | default | |
|---|---|---|
| `--backend` | `torch` | `torch` or `mlx` |
| `--model`, `--revision` | the pinned v2.1 revision | a folder, or a hub id already in the local cache |
| `--execution` | `sequential` | `sequential`: one prompt per forward pass (the published numbers). `batched`: padded microbatches, faster, see below |
| `--dtype`, `--device` | `auto` | torch: bf16 on CUDA, float32 on CPU |
| `--max-prompt-tokens` | 32768 | per question, after the chat template |
| `--max-questions` | 1024 | per request |
| `--host`, `--port` | 127.0.0.1, 8000 | or `$MANCHEGO_HOST`, `$MANCHEGO_PORT` (the Docker image sets 0.0.0.0 and 8000) |
| `--max-queue`, `--queue-timeout` | 32, 600 s | requests allowed to wait for the model, and for how long (then 529) |
| `--no-hash`, `--no-warmup` | off | skip hashing the weight files / the warm-up at start-up |

`--backend`, `--model` and `--revision` default to `$MANCHEGO_BACKEND`, `$MANCHEGO_MODEL` and `$MANCHEGO_REVISION` when set
(the Docker image sets them). Optional bearer-token auth: `MANCHEGO_API_KEYS=key1,key2`; without it every request is accepted.

## The served policy

| | |
|---|---|
| prompt | contract **auto**: questions with at most 26 options use the short prompt Manchego was trained on (question, `State:`, `Options:` with codes A–Z, "Reply with only the letter of the best option."); 27 to 255 options use the state-first prompt of `contract_v2.py` (system message, fenced state, two-letter codes). Chat template with `add_generation_prompt=True`, `enable_thinking=False`. |
| readout | the hidden state at the last prompt position, projected in float32 onto the output-embedding rows of the offered option codes only; softmax at temperature 1.0. Probabilities cover exactly the offered options and sum to 1. |
| passes | one prompt per question, one option order (the client's), no ensembling, no calibration |
| confidence | `(K * max(p) - 1) / (K - 1)`, clamped to [0, 1], K = number of options (TypeSafe's published definition); Noul answers carry none |
| isolation | every question is its own sequence: its own prompt and its own row. No question can attend to, or share state with, another; question names never reach the model |
| execution | `sequential` (default): one forward pass per question, no padding, so a question's numbers do not depend on the rest of the request. `batched` (opt-in): right-padded microbatches of at most 16,384 padded tokens, longest first; still one row per question, but at bf16 or 8 bits the batch shape moves probabilities slightly |

`manchego_serve/contract_v2.py` is the file published with the weights, byte for byte (a test checks its git blob id).

## Pinned weights

| repository | revision (tag `v2.1`) | weight files (SHA-256) | `weights_sha256` reported |
|---|---|---|---|
| `oraculumai/Manchego` (bf16, torch) | `77403228b7dfdf823af99a5f562bdcf80b708d4c` | `model.safetensors-00001-of-00002.safetensors` `1d5df0ff…89efe3`, `-00002-of-00002` `b12ea489…e5aa0a` | `1130745e2a9506ece5a35a1d3da5ad1ce17e8a05052d8fc45d1971986bd8de61` |
| `oraculumai/Manchego-MLX-8bit` | `79e55e2d0c4446abfe0d55d829e8854de9177c1b` | `model.safetensors` `ffa9e0c3…ed5ef43` | `6b0cb89600ffcc0941c557ac60a8445709d5f996a6baec24d9a92213ddccf84b` |
| `oraculumai/Manchego-MLX-4bit` | `184016ce35c3a400880634352360b39cfdbb901e` | `model.safetensors` `301da641…e09e2` | `0fb734ac0221f4fcc314daae3ac2391f1eb41dc2f381ef9c2fe319907e49a98a` |

Full hashes are in `manchego_serve/weights.py`, together with the pinned SHA-256 of each repository's
`chat_template.jinja`, `config.json`, `model.safetensors.index.json`, `tokenizer.json` and `tokenizer_config.json`
(`support_files_verified`: true only when all five are the published files of the same revision as the weights).
`weights_sha256` is the SHA-256 of the lines `"<file sha256>  <file name>\n"` sorted by name. The card's numbers are for
the bf16 weights; the MLX 4-bit build loses accuracy (see the model card).

## API

Request (any `model` string is accepted; extra top-level fields are ignored):

```json
{"model": "manchego-2.1",
 "state": "Customer: the blender I bought last week smells of burning and stopped working. Order 5521.",
 "questions": {
   "route":     {"type": "choice", "instructions": "Which team should handle this message?",
                 "criteria": {"returns": "refunds, exchanges and defective items", "shipping": "delivery status and lost parcels", "billing": null}},
   "defective": {"type": "noul", "instructions": "Does the customer report a defective product?"},
   "urgency":   {"type": "score", "instructions": "How urgent is this message?", "criteria": ["routine", "soon", "immediately"]}}}
```

`state` may be a string or any JSON value (rendered with 2-space indentation). `instructions` and option / level
descriptions may be strings or JSON values. `choice` criteria: an object `{value: description or null}` (2 to 255
entries). `score` criteria: a list of level descriptions (2 to 255). `noul` criteria: optional `{"true": ..., "false": ...}`
descriptions.

Response (this example came from the MLX 8-bit build on Apple silicon; bf16 on CUDA differs in the later digits):

```json
{"model": "manchego-2.1",
 "answers": {
   "route":     {"type": "choice", "choice": "returns", "confidence": 0.997684993793104,
                 "probabilities": {"returns": 0.998456662528736, "shipping": 0.0009666502536993732, "billing": 0.0005766872175646961}},
   "defective": {"type": "noul", "noul": 0.9915093713016409},
   "urgency":   {"type": "score", "score": 1.3388053490887424, "confidence": 0.18063663342085934,
                 "legend": {"0": "routine", "1": "soon", "2": "immediately"},
                 "probabilities": {"0": 0.10371844764867573, "1": 0.45375775561390624, "2": 0.44252379673741804}}},
 "usage": {"input_tokens": 244, "output_tokens": 0},
 "manchego": {"server": "manchego-serve 0.1.0", "backend": "mlx", "precision": "q8g64", "contract": "auto",
              "temperature": 1.0, "permute": 1,
              "confidence_definition": "(K * max(p) - 1) / (K - 1), clamped to [0, 1]; K = number of offered options",
              "execution": "sequential", "model_repo": "oraculumai/Manchego-MLX-8bit",
              "model_revision": "79e55e2d0c4446abfe0d55d829e8854de9177c1b",
              "weights_sha256": "6b0cb89600ffcc0941c557ac60a8445709d5f996a6baec24d9a92213ddccf84b", "weights_verified": true,
              "support_files_verified": true,
              "isolation": "one sequence per question; no question can attend to another",
              "contract_by_question": {"route": "short", "defective": "short", "urgency": "short"}, "forward_passes": 3}}
```

`noul` is P(yes). `score` is the expected level `sum(i * p_i)`. `usage.input_tokens` is the total prompt length over all
questions (each question re-reads the state). `GET /v1/models` lists `manchego-2.1` (alias `manchego-latest`).
`GET /healthz` returns the `manchego` block plus the limits, the weight-file and support-file hashes, the runtime and the
warm-up result.

Errors: `{"detail": {"error_type": ..., "message": ...}}` with HTTP 422 (`invalid_request`), 401 (`authentication_error`,
only with `MANCHEGO_API_KEYS`), 529 (`overloaded_error`, with `retry-after` and `retry-after-ms`) or 500. Nothing is ever
truncated to fit.

## Limits

| limit | value | HTTP 422 message contains |
|---|---|---|
| options per question (choice options, score levels) | 2 to 255 | `options per choice` (too many); `a choice needs at least two options`; `a score takes 2 to 10 levels` (a score with fewer than two levels) |
| prompt length per question, after the chat template | 32,768 tokens | `maximum context length` |
| questions per request | 1,024 | `too many tokens` |

The capacity phrases are among those the Decision Index HTTP engine (`decision_index/engines/http.py`,
`CAPACITY_MARKERS`) treats as "unsupported" rather than as an error; every capacity refusal carries one. Scores take 2
to 255 levels here (the System One contract allows 2 to 10); the official API bounds a request's tokens rather than its
questions, so the question limit is the one refusal with no official counterpart.
The model was trained on prompts up to about 9,000 tokens; longer prompts are accepted up to the limit but untested.
English only. Noul criteria should be strings: under the short prompt a JSON object there is inserted with Python's
`repr`, exactly as in the reference implementation (parity is kept deliberately).

## For benchmark maintainers

**Run recipe (CUDA).**

```bash
git clone <this repository> manchego-serve && cd manchego-serve
docker build -t manchego-serve:2.1-cuda .                      # the only step that uses the network
docker network create --internal manchego-bench                 # a network with no route out
docker run -d --name manchego --gpus all --network manchego-bench manchego-serve:2.1-cuda
# run the harness in the same network, e.g. Decision Index's HTTP engine:
#   DECISION_INDEX_BASE_URL=http://manchego:8000   (any `model` string works)
docker exec manchego python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/healthz').read().decode())"
```

If the harness runs on the host instead, publish the port (`-p 127.0.0.1:8000:8000`); the server still makes no outbound
connection (it forces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1` before any model code
is imported, and never downloads).

- **Record `/healthz` with your results.** It names the weights (`model_repo`, `model_revision`, `weights_sha256`,
  `weights_verified`: true only when the loaded weight files are the published ones for that revision;
  `support_files_verified`: the same for the chat template, tokenizer and config files), the precision, the execution,
  the runtime (torch version, GPU name) and the warm-up.
- **Warm-up** happens at start-up: a fixed invented request is scored twice before the port opens
  (`warmup.first_ms`, `warmup.second_ms`). Start latency measurements after `/healthz` answers.
  `warmup.repeat_identical` says whether the runtime reproduced itself bit for bit.
- **Deterministic settings.** No sampling anywhere: temperature 1.0 over logits, one pass, one option order. Keep the
  default `--execution sequential`: each question is one forward pass with no padding, so its probabilities do not
  depend on which other questions share the request, or on the request's size. (`batched` changes the batch shape
  and, at bf16, moves probabilities in the later digits; it never lets questions see each other.)
  `warmup.repeat_identical` shows whether the runtime repeated itself bit for bit. A different GPU, CPU, driver or
  library version changes the arithmetic, and at bf16 that is not confined to the last digits: the same code at bf16
  on two CPU platforms (the CPU image on linux/arm64, and macOS) differed by up to 0.039 in probability on nine invented
  requests. Compare runs only on the same hardware and image, and report the precision.
- **Kernels.** The image uses Transformers' reference PyTorch kernels for Qwen3.5's linear-attention layers
  (`flash-linear-attention` and `causal-conv1d` are not installed; the log says so at start-up). This is the slower
  path; installing those packages speeds it up and changes the arithmetic slightly.
- **One model worker.** Requests are served one at a time; up to 32 wait (then 529 with `retry-after`), each for at
  most 600 s. Measured on the author's machines (not a CUDA benchmark): MLX 8-bit on an M3 Max about 70 ms per
  short question; the CPU image at bf16 in a 16-CPU VM about 3 s per short question.
- **Refusals.** Over-limit requests are 422 with the capacity phrases above, never truncated or silently shortened.

## Tests

```bash
pip install ".[test]"
pytest                                                   # no weights needed: wire contract, limits, prompt parity
MANCHEGO_TEST_TOKENIZER=/path/to/Manchego pytest         # + token ids and option-code ids
MANCHEGO_TEST_MLX_MODEL=/path/to/Manchego-MLX-8bit pytest -s tests/test_numeric.py     # Apple silicon
MANCHEGO_TEST_TORCH_MODEL=/path/to/Manchego pytest -s tests/test_numeric.py            # informational
```

`tests/fixtures/requests.json` holds 40 invented requests (46 questions: noul, choice with 2, 5, 26, 27, 40, 45, 60,
100 and 255 options, score with 2 to 30 levels, text and JSON states, criteria with and without descriptions, structured
instructions and descriptions). The golden files were produced by the reference implementation (the project's
research server, which implements the served policy; it is not public):

- `golden_prompts.json`: its chat-templated prompt, token ids, contract and option-code token ids for all 46 questions.
  This package reproduces every prompt byte for byte and every id (checked with the tokenizers of both the bf16 and
  the MLX repositories).
- `golden_mlx8bit.json`: its answers for 20 of the requests on `Manchego-MLX-8bit@79e55e2d`, sequential and batched.
  This package's MLX backend reproduces them to within 1e-6 (the observed difference is 0).
- The torch backend is the reference's torch readout (checked identical to it at bf16 on CPU); against the MLX 8-bit
  reference, float32 on CPU is within 0.04 in probability on the 5 requests marked `torch`, with no change of answer.

## Disclosures (from the model card)

- **JevBench-informed.** Manchego's training families were designed from an earlier, unreleased version's per-family
  JevBench hard-tier scores (the published benchmark specification only; no JevBench item text was used). Treat its
  JevBench results with that in mind.
- **Phrase audit.** An early audit used Jev (TypeSafe AI's hosted decision service) to flag phrases in two template data
  pools, and the flagged phrases were dropped. Jev never produced a training label.
- **Selection.** The first training stage's registered checkpoint selection read one development group whose targets
  came from Jev; a rule that excludes that group selects the same checkpoint.
- **Labels.** Every training label was computed by code (exact under the task definition as implemented) or is a
  public dataset's original human annotation. No model-generated label was used.

The model card (<https://huggingface.co/oraculumai/Manchego>) has the results, the training data, the weaknesses and
the data licences.

## Licence

manchego-serve is Apache-2.0 (`LICENSE`, `NOTICE`). `manchego_serve/contract_v2.py` is copied unchanged from the
model repository, also Apache-2.0. The weights are not part of this repository: Manchego v2.1 is a fine-tune of
Qwen3.5-4B (Apache-2.0, Alibaba Cloud), released under Apache-2.0 by oraculumai; an image built with the weights inside
redistributes them under those terms, with the model repository's `LICENSE` and `NOTICE` included in `/models/manchego`.

The interface follows TypeSafe AI's System One wire contract. This project is independent and not affiliated with or
endorsed by TypeSafe AI.
