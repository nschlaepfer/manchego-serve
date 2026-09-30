# manchego-serve

A small, offline HTTP server for **[Manchego](https://huggingface.co/oraculumai/Manchego)**, a 4B decision model,
speaking TypeSafe AI's **System One wire contract**: `POST /v1/systemone`, `GET /v1/models`, `GET /healthz`.

**0.2.0 serves Manchego v3 and Manchego v2.1**, each with its own published serving policy, read from the model
folder's `manchego_config.json` and from the weights' hash ([what changed in 0.2.0](docs/CHANGES-0.2.0.md)):

| | Manchego v3 (the new default) | Manchego v2.1 |
|---|---|---|
| prompt | contract **semif**: SemIf's `direct-options-v1` prompt for 2 to 16 options and a nonempty state, the state-first prompt otherwise | contract **auto**: the short prompt up to 26 options, the state-first prompt for 27 to 255 |
| temperature (the packaged map, applied by weights hash) | choice 1.5, **noul 0.2**, score 1.0 ([Manchego v3](#manchego-v3-020)) | choice 1.791, noul 1.73, score 1.0 ([Temperature map](#temperature-map-012)) |
| on an NVIDIA GPU | CUDA graphs + a lean host path by default: probabilities move by up to 0.0385 on the golden questions, no answer changed ([docs/FAST_PATH.md](docs/FAST_PATH.md)) | the reference path, as in 0.1.x |
| everything else | **one pass** per question, the client's option order, confidence **(K · max p − 1) / (K − 1)**, every question scored as **its own sequence** | the same; 0.1.2's prompts and answers, byte for byte |

The model cards' numbers are at temperature 1.0: `--temperature-map off` reads every model that way. `--no-fast-path`
gives the reference arithmetic. A temperature never changes the chosen option.

> **Manchego v3** is the default (Hub tag `v3`, pinned here by commit): `manchego-serve-download`, the server's default
> model and the Docker image fetch v3. Manchego v2.1 stays available by name (`--revision v2.1`,
> `--build-arg MANCHEGO_VERSION=v2.1`).

The server never opens an outbound connection:
the weights are fetched once at setup (pinned by full commit sha, checked against the pinned SHA-256 of every weight
file and of the chat template, tokenizer and config files) and the server runs with `HF_HUB_OFFLINE=1`. No telemetry.

Runs on Linux + NVIDIA CUDA (PyTorch, bf16), on CPU (PyTorch, float32), and on Apple silicon (MLX, optional).

**Tested so far:**
- the MLX backend on Apple silicon; the PyTorch backend on CPU (macOS, and the CPU image on linux/arm64 with no
  network);
- the **CUDA image on an NVIDIA A10** (24 GB, driver 570.148, 2026-09-23; v0.1.1 serving v2.1 at temperature 1.0):
  built in 76 s including the verified weight download, `/healthz` healthy with weights and support files verified and
  a repeat-identical warm-up, all 40 fixture requests answered (0 argmax changes against the MLX 8-bit reference;
  largest probability difference 0.070, on the 255-option requests, and at most 0.029 elsewhere), a 7,511-token request
  answered in 2.0 s, serial single-question latency p50 81 ms / p95 82 ms (reference linear-attention kernels), peak GPU
  memory 13.5 GB. v0.1.0's Docker build failed on the CUDA base image (PEP 668); v0.1.1 fixes it with one line;
- the **fast path on an NVIDIA RTX 5090** (Windows, 2026-09-30; the v3 release candidate's bf16 weights; the fast-path
  code of 0.2.0, run outside Docker): the lean host path identical to the reference, CUDA graphs within 0.0385 with no
  answer changed, serial single-question p50 124 ms → 26 ms; the v3 temperature map served as a file
  ([docs/FAST_PATH.md](docs/FAST_PATH.md#validation-on-an-rtx-5090-windows)).

Not yet run: a 0.2.0 container image, and 0.2.0's start-up choices on CUDA (the model folder's serving defaults and the
packaged v3 map by hash, tested without CUDA). v0.1.2's v2.1 temperature map has not been run on CUDA (its chosen
options cannot differ, its probabilities do).

## Quick start

### Docker (Linux, NVIDIA GPU)

```bash
docker build -t manchego-serve:3-cuda --build-arg MANCHEGO_VERSION=v3 .      # once v3's revision is pinned (above)
docker build -t manchego-serve:2.1-cuda --build-arg MANCHEGO_VERSION=v2.1 .  # downloads and verifies the weights (9.3 GB) into the image
docker run --rm --gpus all -p 127.0.0.1:8000:8000 manchego-serve:3-cuda
curl -s localhost:8000/healthz
```

`MANCHEGO_VERSION` selects the model: `v3` or `v2.1`. Its default is v2.1 until this package pins v3's revision, then
v3. `MODEL_REVISION` (a full commit sha of `MODEL_REPO`) overrides the pinned commit.

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

Weights outside the image: build with `--build-arg DOWNLOAD_WEIGHTS=0` (and the `MANCHEGO_VERSION` of the folder you
will mount) and mount a folder holding the pinned revision:
`docker run ... -v /path/to/Manchego:/models/manchego:ro manchego-serve:2.1-cuda`. The server hashes what it loads and
reports whether the bytes are the published ones (`weights_verified` for the weight files, `support_files_verified` for
the chat template, tokenizer and config files; in `/healthz` and in every response).

### pip

```bash
git clone <this repository> manchego-serve && cd manchego-serve
# Linux + CUDA: install the CUDA build of torch first
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install .
manchego-serve-download                  # once: the default version's pinned commit of oraculumai/Manchego, verified (uses the network)
manchego-serve --backend torch --host 127.0.0.1 --port 8000     # offline from here on
```

`manchego-serve-download --revision v2.1` (or `v3`) fetches a version by name, and `manchego-serve --revision v2.1`
serves it from the cache: a version name means the commit this package pins, never the Hub's tag. Apple silicon:
`pip install ".[mlx]"`, `manchego-serve-download --repo oraculumai/Manchego-MLX-8bit`, then `manchego-serve --backend mlx`.
`--model` also takes a local folder (`--model /path/to/Manchego --revision <sha or version>`).

| option | default | |
|---|---|---|
| `--backend` | `torch` | `torch` or `mlx` |
| `--model`, `--revision` | the newest pinned version (v2.1 until v3's revision is pinned) | a folder, or a hub id already in the local cache; `--revision` is a full sha or a version name (`v3`, `v2.1`) |
| `--execution` | `sequential` | `sequential`: one prompt per forward pass (the published numbers). `batched`: padded microbatches, faster, see below |
| `--dtype`, `--device` | `auto` | torch: bf16 on CUDA, float32 on CPU |
| `--max-prompt-tokens` | 32768 | per question, after the chat template |
| `--max-questions` | 1024 | per request |
| `--host`, `--port` | 127.0.0.1, 8000 | or `$MANCHEGO_HOST`, `$MANCHEGO_PORT` (the Docker image sets 0.0.0.0 and 8000) |
| `--max-queue`, `--queue-timeout` | 32, 600 s | requests allowed to wait for the model, and for how long (then 529) |
| `--no-hash`, `--no-warmup` | off | skip hashing the weight files / the warm-up at start-up |
| `--temperature-map` | `default` | `default`: the packaged maps, each applied only to the weights it is bound to by hash (v3's to the v3 builds under contract semif, v2.1's to the v2.1 builds it lists under contract auto; T = 1.0 for any other model); `off` (or `--no-temperature-map`): temperature 1.0 for every type, the model cards' numbers; or a JSON file (schema 1, or schema 2 bound to one model's weights). Or `$MANCHEGO_TEMPERATURE_MAP` |
| `--contract` | `model` | `model`: the `contract` field of the model folder's `manchego_config.json`, `auto` when absent (v2.1); `auto`; or `semif`. Or `$MANCHEGO_CONTRACT` |
| `--gdn-kernels` | `reference` | torch, CUDA: `fast` runs flash-linear-attention / causal-conv1d kernels in Qwen3.5's linear-attention layers where importable; `reference` runs Transformers' PyTorch functions even when those packages are installed. Never a model default. Or `$MANCHEGO_GDN_KERNELS`. [docs/FAST_PATH.md](docs/FAST_PATH.md) |
| `--cuda-graphs` / `--no-cuda-graphs`, `--graph-buckets` | the model folder's `serving` field on CUDA (v3: on), else off; `256,512,1024,2048` | torch, CUDA: one-prompt forwards replayed as CUDA graphs at padded lengths. Or `$MANCHEGO_CUDA_GRAPHS=1` / `0`, `$MANCHEGO_GRAPH_BUCKETS` |
| `--fast-host` / `--no-fast-host` | the model folder's `serving` field on CUDA (v3: on), else off | torch: one-prompt calls without needless host syncs; bit for bit the reference. Or `$MANCHEGO_FAST_HOST=1` / `0` |
| `--fast-path` / `--no-fast-path` | neither | all three of the above / none of them (the reference path, whatever the model folder says) |

`--backend`, `--model` and `--revision` default to `$MANCHEGO_BACKEND`, `$MANCHEGO_MODEL` and `$MANCHEGO_REVISION` when set
(the Docker image sets them). Optional bearer-token auth: `MANCHEGO_API_KEYS=key1,key2`; without it every request is accepted.

## The served policy

| | |
|---|---|
| prompt | contract **auto** (Manchego v2.1; the default whenever the model folder's `manchego_config.json` names no contract): questions with at most 26 options use the short prompt Manchego was trained on (question, `State:`, `Options:` with codes A–Z, "Reply with only the letter of the best option."); 27 to 255 options use the state-first prompt of `contract_v2.py` (system message, fenced state, two-letter codes). Chat template with `add_generation_prompt=True`, `enable_thinking=False`. |
| prompt (0.2.0) | contract **semif** (`"contract": "semif"` in `manchego_config.json`, for models trained under SemIf such as Manchego v3): a question with 2 to 16 options and a nonempty state uses SemIf's `direct-options-v1` prompt (`contract_semif.py`: a system message and one JSON user message `{evidence, criterion, options}`, letters A–P, noul as true then false); every other question (17 to 255 options, or a state that is `""`, `{}`, `[]` or `null`) uses the state-first prompt of `contract_v2.py`, never the short prompt. This is the rule the research repository's development reads apply; `tests/test_semif.py` checks every prompt and token id against its training code. `contract_by_question` says which prompt each question got (`semif` or `state_first`). |
| readout | the hidden state at the last prompt position, projected in float32 onto the output-embedding rows of the offered option codes only; softmax at the question type's temperature (the packaged map bound to the weights: v3 choice 1.5, noul 0.2, score 1.0; v2.1 choice 1.791, noul 1.73, score 1.0; any other model, or `--temperature-map off`: 1.0). Probabilities cover exactly the offered options and sum to 1. The chosen option is the largest logit, whatever the temperature. |
| fast path (0.2.0) | the model folder's `serving` field, on CUDA only: v3 declares CUDA graphs + the lean host path, v2.1 nothing. Flags and environment override it. [docs/FAST_PATH.md](docs/FAST_PATH.md) |
| passes | one prompt per question, one option order (the client's), no ensembling; the only calibration is the per-type temperature |
| confidence | `(K * max(p) - 1) / (K - 1)`, clamped to [0, 1], K = number of options (TypeSafe's published definition); Noul answers carry none |
| isolation | every question is its own sequence: its own prompt and its own row. No question can attend to, or share state with, another; question names never reach the model |
| execution | `sequential` (default): one forward pass per question, no padding, so a question's numbers do not depend on the rest of the request. `batched` (opt-in): right-padded microbatches of at most 16,384 padded tokens, longest first; still one row per question, but at bf16 or 8 bits the batch shape moves probabilities slightly |

`manchego_serve/contract_v2.py` is the file published with the weights, byte for byte (a test checks its git blob id).

### Manchego v3 (0.2.0)

Manchego v3 is Qwen3.5-4B with a LoRA adapter trained under the SemIf prompt contract, merged in bf16. Its folder's
`manchego_config.json` says how it is served:

```json
{"model": "Manchego", "version": "v3", "contract": "semif", "served_name": "manchego-3",
 "serving": {"cuda_graphs": true, "fast_host": true}}
```

- **Prompt: contract `semif`** (the table above). v3 was trained on SemIf prompts only. Questions it sees through the
  state-first prompt (17 to 255 options, or an empty state) are outside its training, and so are prompts longer than
  its longest training prompt, 6,386 tokens.
- **Temperature: the packaged map `manchego_serve/temperature_map_v3.json`** (TMAP-V15, schema 2): **choice 1.5,
  noul 0.2, score 1.0**.
  - **What it was fitted on:** v3's SemIf records of 28,182 development rows the model never trained on (choice
    10,485, noul 16,050, score 1,647); no benchmark item. **The rule**, registered before the fit: per type, the
    temperature on a grid from 0.2 to 2.0 that maximises an estimate of JevBench v1.5's composite score, ties going to
    the temperature closest to 1. `/healthz` returns the file's own record (`temperature_map_provenance`).
  - **Bound by hash.** The file names the bf16 weights (`weights_sha256` `2ee83843…`). This package binds the same map
    to the MLX 8-bit (`358b025b…`) and MLX 4-bit (`e1bc5538…`) builds, with the note *"fitted on the bf16 path's
    development records; applied to the MLX builds as-is; chosen options are never changed by a temperature"*, which
    every response carries in `temperature_map.binding`. Other weights, another contract, and `--no-hash` get T = 1.0.
    Passed explicitly as a file, the map binds to the bf16 weights only.
  - **`--temperature-map off`** reads v3 at T = 1.0 for every type: the numbers of v3's model card.
- **Why noul T = 0.2: JevBench v1.5's abstention band.** JevBench v1.5 reads a noul answer as No when P(yes) ≤ 0.20
  and as Yes when P(yes) ≥ 0.80, and scores anything in between as an abstention, which counts as wrong. At T = 1 every
  noul answer whose yes/no logit margin is below ln 4 = 1.39 falls in that band; at T = 0.2 only margins below
  0.2 · ln 4 = 0.28 do.
  - Sharpening never moves an answer across 0.5: it turns would-be abstentions into Yes or No answers, each right or
    wrong as the model's choice is, and turns no Yes into a No.
  - The cost: under the map, P(yes) is **not a calibrated probability**. A margin of 1.0 reads as P(yes) = 0.993
    instead of 0.731, and every margin beyond about 7.35 reads as exactly 1.0 (or 0.0) in double precision. Clients
    that use `noul` as a probability should run with `--temperature-map off`.
  - v2.1's map does the opposite (noul T = 1.73 softens), which is why v2.1 should run with `--temperature-map off` on
    JevBench v1.5 (the warning below).
  - Choice T = 1.5 flattens the choice probabilities (a lower `confidence`); score T = 1.0 leaves scores as they are.
- **Speed: CUDA graphs + the lean host path by default** on an NVIDIA GPU (the folder's `serving` field). On CPU, MPS
  or MLX they are not applied, and `/healthz` says so. On an RTX 5090, serial single-question latency over the golden
  mix went from p50 124 ms / p95 232 ms to p50 26 ms / p95 180 ms.
  - **Disclosed deviation:** the graphs change v3's probabilities by up to 0.0385 on the 83 golden questions (up to
    0.0626 with other bucket sets), with no chosen option changed. That exceeds the 0.03 criterion set before the
    measurement. Serving them by default is the account holder's decision, recorded with the numbers in
    [docs/FAST_PATH.md](docs/FAST_PATH.md#validation-on-an-rtx-5090-windows).
  - `--no-fast-path` (or `--no-cuda-graphs`) serves the reference arithmetic.
  - The flash-linear-attention / causal-conv1d kernels stay off (`--gdn-kernels reference`): they changed an answer.
- **Weights:** see [Pinned weights](#pinned-weights). v3's Hub revisions are filled in after the upload.

### Temperature map (0.1.2)

The probabilities are `softmax(z / T)`, where `z` holds the option-code logits and `T` depends on the question type.
`manchego_serve/temperature_map_v2.1.json` ships with the package and its SHA-256 is pinned in `temperature.py`. It
gives **choice 1.791, noul 1.73, score 1.0**.

> **Warning: the v2.1 map's noul temperature and JevBench v1.5.** JevBench v1.5 reads a noul answer as No when
> P(yes) ≤ 0.20 and as Yes when P(yes) ≥ 0.80, and counts **anything in between as an abstention, scored wrong**. The
> map's noul T = 1.73 *softens* every noul answer: P(yes) ≥ 0.80 then needs a yes/no logit margin of at least
> T · ln 4 = 2.40, instead of 1.39 at T = 1 (the same holds for No). Every noul answer whose margin lies between 1.39 and
> 2.40 is a Yes or a No at T = 1 and an abstention under the map. Under that rule the map can only turn right noul answers
> into wrong ones, never the reverse (a temperature above 1 moves every P(yes) towards 0.5). For v2.1 on JevBench v1.5,
> run with `--temperature-map off`. The map never changes a chosen option; score items are read by their expected
> level, which the map leaves unchanged (score T = 1.0); anything that reads the choice probabilities (a calibration
> measure) sees them flattened.

**0.2.0: T = 1.0 for every model no packaged map is bound to by hash.** The v2.1 map applies only to the v2.1 builds it
lists (by `weights_sha256`), and only under contract `auto`; the v3 map only to the v3 builds, under contract `semif`
([Manchego v3](#manchego-v3-020)). Every other model gets T = 1.0 unless a schema-2 map bound to its weights is passed.
With `--no-hash`, a bare directory with no hub id and no `--revision` now gets T = 1.0 as well (0.1.2 kept the v2.1
map, reported as not checked), and the v3 map is never applied.

**Schema 2 (0.2.0), a map bound to one model.** `--temperature-map /path/map.json` with

```json
{"schema": "manchego-temperature-map/2",
 "temperatures": {"choice": 0.85, "noul": 0.7, "score": 1.0},
 "model_sha256": "<weights_sha256 of the weights it was fitted on, as /healthz reports it>",
 "fitted_on": "which data, which split, which prompt contract",
 "rule": "the objective and the decision rule that chose these values",
 "contract": "semif"}
```

Temperatures may be below 1 (sharpening) or above (flattening), each in [0.05, 20]. `fitted_on` and `rule` are required
text; `contract` is optional. The server refuses to start when `model_sha256` is not the loaded weights' SHA-256, when
the weights were not hashed (`--no-hash`), or when `contract` is not the served contract. Such a map is never applied
"as asked". `/healthz` reports the whole file under `temperature_map_provenance`.

- **What it changes.** `noul`, `probabilities`, `confidence` and a score's expected level all change. The chosen option
  never changes, because it is read from the logits. Accuracy is therefore identical.
- **What it was fitted on.** Held-out development records of v2.1 only, in a protocol registered before the fit:
  - the Stage-0 calibration draws of the project's hard-gap screen (generated long policies, multi-hop lookups, judging
    of worked responses, and fresh draws of v2.1's own families);
  - 22 Natural Instructions tasks that v2.1 never trained on.

  It was fitted on half of the instances (or tasks) and evaluated on the other half. **Never used:** any JevBench item or
  record, Public8, test sets, sealed sets, and model-generated labels.
- **Held-out result** (3,524 rows, accuracy 0.639, which T does not change):

  | measure | T = 1.0 | map | change [95% interval] |
  |---|---:|---:|---:|
  | NLL | 0.832 | 0.767 | −0.065 [−0.082, −0.048] |
  | ECE, 15 bins | 0.093 | 0.018 | −0.075 [−0.081, −0.046] |
  | Brier | 0.465 | 0.445 | −0.019 [−0.025, −0.013] |
  | distance to a one-hot target (TVD) | 0.407 | 0.443 | **+0.036** [+0.032, +0.040] |

  - The gain comes from the hard components: ECE goes from 0.15 to 0.07 on long policies, 0.12 to 0.08 on judging (not
    resolved on its own) and 0.15 to 0.08 on multi-hop.
  - On the 22 unseen NI tasks it is unresolved (ECE 0.075 to 0.060).
  - One temperature shared by all types does just as well: the per-type map adds nothing measurable.
- **What it costs.** v2.1 is already well calibrated at T = 1 on the task types it was trained on, and the map
  over-flattens them. On its own development groups (2,920 rows, accuracy 0.885, reported only, never fitted):
  - ECE goes from 0.023 to 0.092 and NLL rises by 0.055 [0.049, 0.061];
  - under the map every such group is under-confident.

  TVD rises everywhere, because flatter probabilities put less mass on the right answer.
- **Not covered by the fit.**
  - Every fitted and held-out row used the short prompt (up to 26 options). On 27 to 255 options the map is unmeasured;
    on the familiar large-menu group (reported only) ECE goes from 0.005 to 0.080.
  - The fit set has no soft (gold-distribution) targets. On the 102 such choice and noul rows of the familiar groups
    (reported only), the distance to the gold distribution rises from 0.155 to 0.197.
- **Which builds.**
  - The map is applied to the bf16 weights (`oraculumai/Manchego`). It was fitted on the same weights in bf16 on CUDA,
    with the adapter unmerged.
  - It is also applied to the MLX 8-bit build. On 321 held-out rows scored with the same prompt token ids, the
    temperature fitted on that build's logits came within 2% of the one fitted on the bf16 logits (2.60 vs 2.64 for
    that harder sample). The two builds chose the same option on 98.1% of those rows.
  - It is not applied to the MLX 4-bit build or to any other hashed weights (T = 1.0, with a warning), unless a map
    file is passed explicitly. With `--no-hash` the declared hub id and revision decide: a build the map does not list
    gets T = 1.0; a listed one, or a bare directory with nothing declared, keeps the map, reported as not checked.
  - float32 on CPU is unmeasured.
- **Reported.**
  - `/healthz` and every response carry `temperature` (1.0 when off, else the map), `temperature_map` (source, SHA-256,
    binding to the loaded weights) and `temperature_by_question`.
  - `/healthz` also returns the file's own provenance (`temperature_map_provenance`).
- **Turning it off.** `--temperature-map off`, `--no-temperature-map` or `MANCHEGO_TEMPERATURE_MAP=off`. This is the
  v0.1.0 readout (unchanged in v0.1.1), byte for byte; a test checks it against the v0.1.0 golden fixtures.

## Pinned weights

| repository | version | revision | weight files (SHA-256) | `weights_sha256` reported |
|---|---|---|---|---|
| `oraculumai/Manchego` (bf16, torch) | v3 | `f82e029d0ad4d1bdd1ca12f5f7b548b84fabc9a9` | `model.safetensors-00001-of-00002.safetensors` `036f8c81…86b89d`, `-00002-of-00002` `44fb326e…428af6` | `2ee838433bfe278a226dc644667ad4a99ece82cc47325c7645a7dae723c1863b` |
| `oraculumai/Manchego-MLX-8bit` | v3 | `4ebdc0dd50481cfa9e40f83f05571ee59e53041d` | `model.safetensors` `02546297…248e85` | `358b025b04001e50a065f8c87929175211264bd6182af74b67bd6caa2f639657` |
| `oraculumai/Manchego-MLX-4bit` | v3 | `0c17e076e8e358e41a9f8332dca062e4942ace0b` | `model.safetensors` `a6e8ea1e…8428be` | `e1bc5538b8dced2a857b4980dba045c2ca01db1c369aa416fc19f0f5c593e782` |
| `oraculumai/Manchego` (bf16, torch) | v2.1 | `77403228b7dfdf823af99a5f562bdcf80b708d4c` (tag `v2.1`) | `model.safetensors-00001-of-00002.safetensors` `1d5df0ff…89efe3`, `-00002-of-00002` `b12ea489…e5aa0a` | `1130745e2a9506ece5a35a1d3da5ad1ce17e8a05052d8fc45d1971986bd8de61` |
| `oraculumai/Manchego-MLX-8bit` | v2.1 | `79e55e2d0c4446abfe0d55d829e8854de9177c1b` (tag `v2.1`) | `model.safetensors` `ffa9e0c3…ed5ef43` | `6b0cb89600ffcc0941c557ac60a8445709d5f996a6baec24d9a92213ddccf84b` |
| `oraculumai/Manchego-MLX-4bit` | v2.1 | `184016ce35c3a400880634352360b39cfdbb901e` (tag `v2.1`) | `model.safetensors` `301da641…e09e2` | `0fb734ac0221f4fcc314daae3ac2391f1eb41dc2f381ef9c2fe319907e49a98a` |

v3's SHA-256 were computed from the release builds that are uploaded; its support files are v2.1's, byte for byte. A
v3 revision counts as pinned only once it is a full commit sha: until then v3 weights are reported as matching no
pinned revision (`weights_verified: false`), and they still get v3's serving policy, which follows the folder's
`manchego_config.json` and the weights' hash.

Full hashes are in `manchego_serve/weights.py`, together with the pinned SHA-256 of each repository's
`chat_template.jinja`, `config.json`, `model.safetensors.index.json`, `tokenizer.json` and `tokenizer_config.json`
(`support_files_verified`: true only when all five are the published files of the same revision as the weights).
`weights_sha256` is the SHA-256 of the lines `"<file sha256>  <file name>\n"` sorted by name. The card's numbers are for
the bf16 weights; the MLX builds' own numbers are in their model cards (v2.1's MLX 4-bit build loses accuracy).

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

Response (this example came from Manchego v2.1's MLX 8-bit build on Apple silicon with its default temperature map,
recorded with 0.1.2; 0.2.0 gives the same answers and says `manchego-serve 0.2.0`; bf16 on CUDA differs in the later
digits):

```json
{"model": "manchego-2.1",
 "answers": {
   "route":     {"type": "choice", "choice": "returns", "confidence": 0.9474451113245261,
                 "probabilities": {"returns": 0.964963407549684, "shipping": 0.020027128735617024, "billing": 0.01500946371469903}},
   "defective": {"type": "noul", "noul": 0.9400035618635341},
   "urgency":   {"type": "score", "score": 1.3388053490887424, "confidence": 0.18063663342085934,
                 "legend": {"0": "routine", "1": "soon", "2": "immediately"},
                 "probabilities": {"0": 0.10371844764867573, "1": 0.45375775561390624, "2": 0.44252379673741804}}},
 "usage": {"input_tokens": 244, "output_tokens": 0},
 "manchego": {"server": "manchego-serve 0.1.2", "backend": "mlx", "precision": "q8g64", "contract": "auto",
              "temperature": {"choice": 1.791, "noul": 1.73, "score": 1.0},
              "temperature_map": {"source": "default", "temperatures": {"choice": 1.791, "noul": 1.73, "score": 1.0},
                                  "file": "temperature_map_v2.1.json",
                                  "sha256": "2458d21771bbddc9c23ae0e63c880f2ebe5a6aa997e8c9de6b5e67b7210cd07e",
                                  "fitted_for": "Manchego v2.1",
                                  "binding": "a build this map lists: oraculumai/Manchego-MLX-8bit@79e55e2d0c4446abfe0d55d829e8854de9177c1b"},
              "permute": 1,
              "confidence_definition": "(K * max(p) - 1) / (K - 1), clamped to [0, 1]; K = number of offered options",
              "execution": "sequential", "model_repo": "oraculumai/Manchego-MLX-8bit",
              "model_revision": "79e55e2d0c4446abfe0d55d829e8854de9177c1b",
              "weights_sha256": "6b0cb89600ffcc0941c557ac60a8445709d5f996a6baec24d9a92213ddccf84b", "weights_verified": true,
              "support_files_verified": true,
              "isolation": "one sequence per question; no question can attend to another",
              "contract_by_question": {"route": "short", "defective": "short", "urgency": "short"},
              "temperature_by_question": {"route": 1.791, "defective": 1.73, "urgency": 1.0}, "forward_passes": 3}}
```

With `--temperature-map off` (T = 1.0, as in v0.1.0) the same request gives route `returns` with p = 0.998456662528736
(confidence 0.997684993793104), `defective` 0.9915093713016409 and the same score. The choice is the same either way.

`noul` is P(yes). `score` is the expected level `sum(i * p_i)`. `usage.input_tokens` is the total prompt length over all
questions (each question re-reads the state). `GET /v1/models` lists the served model's name (`served_name` in its `manchego_config.json`: `manchego-3` for v3;
`manchego-2.1` when absent) and the alias `manchego-latest`.
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
docker build -t manchego-serve:3-cuda --build-arg MANCHEGO_VERSION=v3 .   # the only step that uses the network
docker network create --internal manchego-bench                 # a network with no route out
docker run -d --name manchego --gpus all --network manchego-bench manchego-serve:3-cuda
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
- **Deterministic settings.** No sampling anywhere: one fixed temperature per question type over the logits (reported
  in `/healthz` and in every response as `temperature`, `temperature_map` and `temperature_by_question`; `off` gives
  1.0, the v0.1.0 policy), one pass, one option order. Keep the
  default `--execution sequential`: each question is one forward pass with no padding, so its probabilities do not
  depend on which other questions share the request, or on the request's size. (`batched` changes the batch shape
  and, at bf16, moves probabilities in the later digits; it never lets questions see each other.)
  The temperature is the packaged map bound to the weights (v3: choice 1.5, noul 0.2, score 1.0; v2.1: see above);
  `--temperature-map off` gives 1.0. `warmup.repeat_identical` shows whether the runtime repeated itself bit for bit.
  A different GPU, CPU, driver or
  library version changes the arithmetic, and at bf16 that is not confined to the last digits: the same code at bf16
  on two CPU platforms (the CPU image on linux/arm64, and macOS) differed by up to 0.039 in probability on nine invented
  requests. Compare runs only on the same hardware and image, and report the precision.
- **Kernels and the fast path.** The image uses Transformers' reference PyTorch kernels for Qwen3.5's linear-attention
  layers (`flash-linear-attention` and `causal-conv1d` are not installed; the log says so at start-up). Since 0.2.0 the
  reference kernels run even when those packages are installed; `--gdn-kernels fast` (or `--fast-path`) opts in to
  them and changes the arithmetic slightly. **Manchego v3 on CUDA runs CUDA graphs and the lean host path by default**
  (its `manchego_config.json`), which moves its probabilities by up to 0.0385 on the golden questions and changes no
  answer ([docs/FAST_PATH.md](docs/FAST_PATH.md#validation-on-an-rtx-5090-windows)); `--no-fast-path` gives the
  reference arithmetic. `/healthz` reports what runs under `runtime.gdn_kernels` and `fast_path_settings`, and every
  response carries `manchego.fast_path` when a fast-path piece is on.
- **One model worker.** Requests are served one at a time; up to 32 wait (then 529 with `retry-after`), each for at
  most 600 s. Measured on the author's machines (not a CUDA benchmark): MLX 8-bit on an M3 Max about 70 ms per
  short question; the CPU image at bf16 in a 16-CPU VM about 3 s per short question.
- **Refusals.** Over-limit requests are 422 with the capacity phrases above, never truncated or silently shortened.

## Tests

```bash
pip install ".[test]"
pytest                                                   # no weights needed: wire contract, limits, prompt parity, temperature map,
                                                         # the SemIf contract, the fast path (with torch: a tiny random model on CPU)
MANCHEGO_TEST_TOKENIZER=/path/to/Manchego pytest         # + token ids and option-code ids (both contracts)
MANCHEGO_TEST_MLX_MODEL=/path/to/Manchego-MLX-8bit pytest -s tests/test_numeric.py     # Apple silicon
MANCHEGO_TEST_TORCH_MODEL=/path/to/Manchego pytest -s tests/test_numeric.py            # informational
```

Without weights the suite also covers the model folder's serving defaults (a tiny random Qwen3.5 on CPU, with CUDA
stood in for), the packaged v3 map and its bindings, the v3 pins with their placeholder revisions, the download's
version names, and the Dockerfile's default version.

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
- `golden_logits_mlx8bit.json` (0.1.2) holds the option-code logits behind `golden_mlx8bit.json`, recorded with this
  package's MLX backend (`tests/record_golden_logits.py`). It lets `tests/test_temperature.py` check, without weights,
  three things:
  - `--temperature-map off` reproduces the v0.1.0 golden answers byte for byte (`json.dumps` equal);
  - the default map never changes a chosen option;
  - the map loads, is the pinned file, binds to the loaded weights and applies per question type.

  `tests/v010_reference.py` is v0.1.0's readout, frozen.
- `golden_semif.json` (0.2.0) holds, for the 37 invented questions of `semif_requests.json`, the training code's
  rendering under contract semif: messages, codes, slot order, chat-templated prompt, token ids and option-code ids. 29
  are SemIf prompts and 8 are the contract v2 overflow. It was recorded by `tests/record_semif_golden.py` from the
  research repository alone, using a tokenizer and no weights. Three training paths agree on it: the renderer, the lean
  trainer's encoder, and the development reads' `encode()`. `tests/test_semif.py` checks every prompt byte for byte, and
  every id when a tokenizer is available. It also checks that `contract_semif.py`'s code (its source without docstrings,
  comments and blank lines) is the training renderer's, by a recorded SHA-256; the documentation was written for
  publication.

## Disclosures (from Manchego v2.1's model card)

Manchego v3's disclosures are in its own model card (<https://huggingface.co/oraculumai/Manchego>).

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

**Serving addition (0.1.2): the v2.1 temperature map.**
- It was fitted on development data that includes the hard-gap components. Their families were designed from the
  published JevBench hard-tier specification (families only; no JevBench item text).
- No JevBench item, per-item result or record was used to fit it or check it.
- It is not part of the model card's numbers, which are at temperature 1.0.

**Serving additions (0.2.0), Manchego v3.**
- The v3 temperature map was chosen to maximise an estimate of JevBench v1.5's composite score on development rows the
  model never trained on; no JevBench item was used. Its noul temperature makes P(yes) deliberately uncalibrated
  ([Manchego v3](#manchego-v3-020)). It is not part of the model card's numbers, which are at temperature 1.0.
- v3's default fast path on CUDA moves its probabilities by up to 0.0385 on the golden questions, more than the 0.03
  criterion set beforehand, with no answer changed: the account holder's disclosed decision
  ([docs/FAST_PATH.md](docs/FAST_PATH.md#validation-on-an-rtx-5090-windows)).

## Licence

manchego-serve is Apache-2.0 (`LICENSE`, `NOTICE`). `manchego_serve/contract_v2.py` is copied unchanged from the
model repository, also Apache-2.0. `manchego_serve/contract_semif.py` renders SemIf's `direct-options-v1` prompt and
includes SemIf's system message text: SemIf (formerly OpenJev, <https://github.com/TheoLeeCJ/SemIf>) is MIT-licensed,
and `NOTICE` carries its licence. The weights are not part of this repository: Manchego v3 and v2.1 are fine-tunes of
Qwen3.5-4B (Apache-2.0, Alibaba Cloud), released under Apache-2.0 by oraculumai; an image built with the weights inside
redistributes them under those terms, with the model repository's `LICENSE` and `NOTICE` included in `/models/manchego`.

The interface follows TypeSafe AI's System One wire contract. This project is independent and not affiliated with or
endorsed by TypeSafe AI.
