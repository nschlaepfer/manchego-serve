# The fast path (0.2.0): what it is, how it was validated, and how to validate it again

Every piece is **off** unless the command line, the environment or the model folder turns it on. A model folder turns
pieces on with a `serving` field in its `manchego_config.json`, and only for the torch backend on CUDA. **Manchego v3's
folder declares `"serving": {"cuda_graphs": true, "fast_host": true}`**, so on an NVIDIA GPU v3 is served with CUDA
graphs and the lean host path by default. The linear-attention kernels stay `reference`. This default changes v3's
probabilities by up to 0.0385 on the golden questions and changes no answer; that exceeds the 0.03 criterion set
beforehand, and it was accepted as a disclosed decision of the account holder
([Validation on an RTX 5090](#validation-on-an-rtx-5090-windows)). Manchego v2.1's folder declares nothing: v2.1 runs
the 0.1.x code, the published numbers' arithmetic, unless a flag asks otherwise.

| flag | what it does | arithmetic | device |
|---|---|---|---|
| `--gdn-kernels reference` (default) | Qwen3.5's gated-delta-net (linear-attention) layers run Transformers' reference PyTorch functions, **even when** flash-linear-attention or causal-conv1d is installed. (0.1.x used those packages silently whenever Transformers could import them.) Never a model default. | the published path | any |
| `--gdn-kernels fast` | fla's `chunk_gated_delta_rule` (and `fused_recurrent_gated_delta_rule`) and causal-conv1d's `causal_conv1d_fn` (and `causal_conv1d_update`), each where importable. `/healthz` → `runtime.gdn_kernels` names what runs for each function. | changes (bf16 kernel arithmetic) | CUDA |
| `--cuda-graphs` / `--no-cuda-graphs` (`--graph-buckets 256,512,1024,2048`) | Each one-prompt forward is replayed as a CUDA graph captured at start-up, at the smallest padded length that holds the prompt. There is no attention mask and no KV cache, and the hidden state is read at the prompt's own last position. Longer prompts and padded batches run eagerly. | changes slightly: the padding cannot reach the answer (every layer is causal, the padding comes after the read position), but the kernels see another shape | CUDA |
| `--fast-host` / `--no-fast-host` | The eager one-prompt path without host work it does not need. Ids and option-code ids go to the device in one pinned non-blocking copy each, and there is no all-ones attention mask (Transformers checks one on the host twice per forward, then drops it). Option-code rows of the output matrix are cached per code set, and there is one device-to-host copy per question. | **bit for bit** the reference | any |
| `--fast-path` / `--no-fast-path` | all three / none of them (the reference path, whatever the model folder or the environment says) | changes / the published path | CUDA / any |

Environment variables: `MANCHEGO_GDN_KERNELS=fast`, `MANCHEGO_CUDA_GRAPHS=1` or `0`, `MANCHEGO_GRAPH_BUCKETS=...` and
`MANCHEGO_FAST_HOST=1` or `0`. Responses carry `manchego.fast_path` only when a piece is on.

## How the default is chosen (0.2.0)

- **Per setting, the first source that says something wins:** the command line, then the environment, then the model
  folder's `serving` field, then off. `--no-cuda-graphs`, `--no-fast-host` and `--no-fast-path` (or
  `MANCHEGO_CUDA_GRAPHS=0`, `MANCHEGO_FAST_HOST=0`) always turn a piece off.
- **`serving` is read only by the torch backend on a CUDA device.** Elsewhere (CPU, MPS, the MLX backend) it is not
  applied, the server starts on the reference path, and `/healthz` says so in `fast_path_settings.note`.
- **The fields:** `cuda_graphs` and `fast_host` (true or false), and an optional `graph_buckets` (a list of distinct
  positive token counts). Any other key, a `gdn_kernels` key included, stops the server at start-up: the kernels are
  chosen by `--gdn-kernels` only.
- **Reported:** `/healthz` → `fast_path_settings` gives what is in force, where each setting came from
  (`command line or environment`, `manchego_config.json (serving)` or `default`), the model's defaults and which of
  them were applied. It and `model_config.serving` appear only when the folder declares `serving` or a piece is on, so
  a folder without the field (v2.1) gets exactly 0.1.x's `/healthz`.
- **Failure:** if a fast path taken from the model folder cannot start (a capture error, for example), the server does
  not start; the log names the flags that turn it off.

## Profiling on the Mac (Apple M3 Max; no CUDA)

The stand-in was the local Manchego v3 release candidate (`v3-100m-b-rc1-release`, bf16 weights), with the torch
backend and the reference kernels. All numbers are informational only: MPS and CPU are not the benchmark path.

- **Host-side preparation is not the bottleneck.** Per question, the chat template takes 0.05 ms and tokenisation
  0.2 to 0.3 ms (46 fixture questions, median prompt 91 tokens). Transformers compiles the Jinja template once and
  caches it, so nothing is rebuilt per request. `--fast-host` therefore targets host syncs and small transfers.
- **MPS, bf16, reference kernels: forward time scales with the prompt length.**

  | prompt tokens | 128 | 256 | 512 | 1024 | 2048 |
  |---|---:|---:|---:|---:|---:|
  | forward (ms) | 353 | 724 | 1412 | 2838 | 6217 |

  That is about 2.8 ms per token. Host-to-device copies took about 1 ms and the readout 1 to 23 ms. With a sync
  around each module at 256 tokens, the reference chunked gated delta rule took 61% of the forward, the MLPs 17%, full
  attention 3% and the depthwise conv 2%. This is the part `--gdn-kernels fast` replaces on CUDA.
- **CPU, float32: the depthwise `F.conv1d` dominates.** It took 88% of the forward: `F.conv1d` with 8,192 groups
  took 324 ms per call at 256 tokens, against 1.1 ms for an equivalent shift-and-add. The two differ by at most 5e-7.
  This is a CPU-only finding, outside this change. On CUDA, cuDNN or causal-conv1d handles this layer.
- **No mask vs an all-ones mask:** bit-identical hidden states (MPS bf16, real weights), as the Transformers code says.
- **Right padding (one question, 91 → 512 tokens, MPS bf16):** the hidden state moved by up to 0.28, the option
  logits by up to 0.06, and the probabilities by up to 0.00025.
- **`agree` preview on MPS** (`tools/fast_path_check.py agree --contract semif --device mps --dtype bfloat16
  --configs fast_host,graphs,kernels --emulate-graphs`): the 83 golden questions under SemIf, prompts of 97 to 4,783
  tokens.
  - `fast_host`: max |Δp| **exactly 0**, no chosen option changed.
  - `graphs`, with the eager stand-in for the capture (so this measures the padding only): max |Δp| **0.0171**, on
    `semif/instructions_whitespace`, with no chosen option changed. 84 calls went through a bucket and 2 ran eagerly
    (the 2 prompts over 2,048 tokens).
  - `kernels`: refused, as designed: it needs CUDA.
  - Padding alone used more than half of the 0.03 budget on one question. On CUDA the graphs alone exceeded it
    ([Validation on an RTX 5090](#validation-on-an-rtx-5090-windows)).
  - An earlier run of the same preview found `fast_host` far off: max |Δp| 0.96 and 48 changed choices. The cause was a
    non-blocking copy from a temporary pageable CPU tensor, which on MPS reads freed memory every time. It is fixed (the
    fast path copies non-blocking only from pinned memory, that is on CUDA) and covered by a test that runs on MPS.

## Validation on an RTX 5090 (Windows)

Run on 2026-09-30 on the project's RTX 5090, with the tools of the next section.

- **Machine:** NVIDIA GeForce RTX 5090 (32,607 MiB), driver 617.14, Windows. torch 2.10 (CUDA 12.8 build), bf16.
  Triton 3.8.0 and flash-linear-attention 0.5.2 installed; causal-conv1d not installed.
- **Model:** the Manchego v3 release candidate rc1, merged bf16 weights (`weights_sha256` `2ee83843…`), in a folder
  whose `manchego_config.json` selects contract `semif`.
- **Code:** manchego-serve 0.2.0.dev0 (`be64e5b`). The fast-path code that ran is 0.2.0's; 0.2.0 adds only how the
  default is chosen. The server ran directly (`python -m manchego_serve`, not in Docker), one server at a time, on an
  otherwise idle GPU.

**Agreement** (`tools/fast_path_check.py agree`; the 83 golden questions, one question per forward pass, the same
loaded model; max |Δp| over every option of every question at T = 1):

| configuration | graph buckets | max \|Δp\| | on question | chosen option changed | against the 0.03 criterion |
|---|---|---:|---|---:|---|
| `fast_host` | — | **0** (identical) | — | 0 | passes (bit for bit) |
| `graphs` | 256, 512, 1024, 2048 (the default) | **0.0385** | `semif/instructions_whitespace` | 0 | exceeds |
| `graphs` | 15 buckets: 128 to 512 by 64, 640 to 1024 by 128, 1280 to 2048 by 256 | 0.0305 | `choice_numeric_keys/q` | 0 | exceeds |
| `graphs` | 29 buckets: 64 to 512 by 32, 576 to 1024 by 64, 1152, 1280, 1408, 1536, 1792, 2048 | **0.0626** | `whitespace_state/q` | 0 | exceeds |
| `kernels` (fla) | — | 0.0562 | `choice255_desc/q` | **1** (`choice27_json/q`, reference top-2 logit gap 0.043) | fails: rejected |

- With graphs, 84 of the 86 forward passes (the 83 questions and a 3-question warm-up) went through a graph; the 2
  prompts over 2,048 tokens ran eagerly.
- Finer buckets do not shrink the shift steadily (0.0305 with 15 buckets, 0.0626 with 29): bf16 kernels accumulate in
  a different order at a different sequence length, so any padded shape moves the numbers. No bucket set measured
  stayed within 0.03.
- **The kernels row comes from an earlier run of the same day** in which the model folder's `manchego_config.json` was
  not yet in place, so those questions were rendered under contract `auto` (v2.1's prompts), not `semif`. Only fla's
  gated delta rule was replaced (causal-conv1d was not installed). The same run measured `graphs` at 0.0220 under
  `auto`. The kernels were rejected on the changed answer and were not measured again under `semif`.
- `agree` exited with status 1 for the graphs configurations, as designed. Its threshold was not changed.

**Latency** (`tools/fast_path_check.py latency`: 450 serial single-decision requests cycling through the 83 golden
questions, after 20 warm-up requests; mean 347 input tokens; no errors; milliseconds):

| server flags | p50 | p95 | mean of p50 and p95 |
|---|---:|---:|---:|
| none (the reference path; the folder had no `serving` field then) | 124.45 | 232.02 | 178.23 |
| `--cuda-graphs --fast-host` | **25.59** | **180.34** | **102.96** |
| `--cuda-graphs --fast-host`, with the v3 temperature map passed as a file | 35.03 | 181.13 | 108.08 |

| prompt tokens | requests | reference p50 / p95 | graphs + fast host p50 / p95 |
|---|---:|---:|---:|
| up to 256 | 339 | 114.93 / 136.30 | 24.75 / 47.89 |
| 257 to 512 | 60 | 141.54 / 164.32 | 43.56 / 66.05 |
| 513 to 1,024 | 28 | in the raw files | in the raw files |
| over 1,024 | 23 | in the raw files | in the raw files |

- Graphs + fast host cut p50 by 4.9× (4.6× for prompts up to 256 tokens, 3.2× for 257 to 512), and the mean of p50 and
  p95 by 1.7×.
- **p95 is set by the mix, not by short prompts.** 23 of the 450 requests (5%) have prompts over 1,024 tokens, and the
  nearest-rank p95 is the 428th of 450 samples: it sits at the edge of that long tail. For short single decisions read
  the by-length rows.
- **The temperature map costs nothing measurable.** It divides a few logits on the host. The by-length rows of the two
  graph runs agree within 0.6 ms. The overall p50 moved from 25.59 to 35.03 ms because the median of this mix falls
  where few samples are: the up-to-256 row alone spans 24.3 to 49.9 ms with its own median at 24.75 ms.
- The rows for longer prompts, and GPU memory with graphs, are in the run's raw files and are not reproduced here.

**Decision (the account holder's, 2026-09-30; a disclosed decision, not a passed test).** The criterion set before
the run was max |Δp| ≤ 0.03 with no chosen option changed. CUDA graphs exceeded it: max |Δp| **0.0385** at the default
buckets, and up to **0.0626** across the bucket sets measured, with **no chosen option changed** in any of them. The
account holder accepted **CUDA graphs + fast host as Manchego v3's serving default** (in v3's `manchego_config.json`),
with this disclosure, and **rejected the fla / causal-conv1d kernels**, which stay off. For scale, the reference path
itself differs across runtimes by similar amounts: 0.039 in probability between two CPU platforms at bf16 (nine
invented requests), and up to 0.070 between CUDA bf16 and the MLX 8-bit build on 255-option requests (at most 0.029
elsewhere). That comparison is context, not a validation: the graphs' shift adds to whichever runtime serves the model.
`--no-cuda-graphs` (or `--no-fast-path`) serves v3 on the reference path.

## Validation on an NVIDIA machine

Record every output file, `nvidia-smi`, and the package versions printed in step 2.

### 1. Environment

```bash
git clone <this repository> manchego-serve && cd manchego-serve && git checkout v0.2.0
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install ".[test]"
pytest -q                          # the weight-free suite; tests/test_fast_path.py runs on CPU even here
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee gpu.txt
```

### 2. Kernels (only for `--gdn-kernels fast` and `--fast-path`)

```bash
pip install flash-linear-attention
pip install causal-conv1d --no-build-isolation      # compiles against the installed torch if no wheel matches (needs nvcc 12.8)
python - <<'EOF' | tee versions.txt
import torch, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__, torch.cuda.get_device_name())
for m in ("fla", "causal_conv1d"):
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "NOT importable:", type(e).__name__, e)
EOF
```

**Hopper (H100, GH200).** This project's earlier cloud runs saw flash-linear-attention 0.5.2 refuse Triton 3.4.0 to
3.7.0 on Hopper, because of a gradient-kernel bug (fla issue #640). torch 2.10.0 pins Triton 3.6.0. There, install
`pip install "triton>=3.7.1"` (pip warns about torch's pin; nothing here uses `torch.compile`). Leave Triton alone on
Ampere and Ada (A10, L4, RTX 4090). A kernel that cannot be imported is reported as `reference (... not importable: ...)`
in `runtime.gdn_kernels` and in the `agree` report; nothing falls back silently.

### 3. Weights

You need a local model folder: the weights, the tokenizer and chat template files, and a `manchego_config.json`. For a
model trained under SemIf (Manchego v3), the file must contain at least:

```json
{"model": "Manchego", "version": "v3", "contract": "semif", "served_name": "manchego-3",
 "serving": {"cuda_graphs": true, "fast_host": true}}
```

The published v2.1 folder has no `contract` and no `serving` field, which means `auto` and the reference path. Check
both models if both will be served. The server never downloads anything (`manchego-serve-download` fetches a pinned
revision once: `--revision v3` or `--revision v2.1`).

### 4. Numerical agreement against the reference path (golden requests, one loaded model)

```bash
python tools/fast_path_check.py agree --model /models/manchego-v3 --out agree_v3.json
python tools/fast_path_check.py agree --model /models/manchego    --out agree_v21.json     # v2.1, contract auto
```

The tool loads the model once in bf16 and scores all 83 golden questions with the reference path, one question per
forward pass, under the model's contract (it never applies the folder's `serving` defaults to its reference). The questions are the 46 of `tests/fixtures/requests.json` and the 37 of
`semif_requests.json`; they are invented, and the mix includes 255-option menus and one 4,783-token prompt. It then
switches the same model to each configuration (`fast_host`, `kernels`, `graphs`, `all`) and scores them again.

**Pass criteria** (the exit status is 1 on any failure):

| configuration | max \|Δp\| over every option of every question (T = 1) | chosen option |
|---|---|---|
| `fast_host` | **exactly 0** | identical on every question |
| `kernels`, `graphs`, `all` | **≤ 0.03** (`--max-dprob`) | identical on every question |

Near ties are not exempt by default. If `--near-tie` is used, report its value and the listed questions. Why 0.03: the
bf16 reference path itself moves by that much between runtimes. It differed by up to 0.039 between two CPU platforms,
and CUDA bf16 against the MLX 8-bit build differed by up to 0.029 (up to 0.070 on 255 options). A fast path should stay
inside that envelope, and never change a choice.

Also read these fields in the report:
- `configs.graphs.calls`: how many questions went through a graph and how many ran eagerly (prompts over 2,048 tokens).
- `configs.*.fast_path.graphs`: the captured buckets.
- `configs.*.error`: a configuration that failed to set up (for example a capture error) fails, and is not skipped.
- `configs.*.latency`: per-question forward latency on this set. That is informational, not the benchmark measure.

### 5. Latency: 450 serial single-decision requests

Run each configuration on an otherwise idle GPU, one server at a time. Start the latency client only after `/healthz`
answers; the server warms up before opening the port.

```bash
M=/models/manchego-v3
# every run names its configuration: a v3 folder turns graphs + fast host on by itself (manchego_config.json, serving)
run() {   # $1 = output name, then server flags
  name=$1; shift
  manchego-serve --model "$M" --port 8000 "$@" > "server_$name.log" 2>&1 &
  pid=$!
  until curl -sf 127.0.0.1:8000/healthz > "healthz_$name.json"; do
    kill -0 $pid 2>/dev/null || { echo "server '$name' exited:"; tail -20 "server_$name.log"; return 1; }
    sleep 5
  done
  python tools/fast_path_check.py latency --url http://127.0.0.1:8000 --n 450 --warmup 20 --out "latency_$name.json"
  kill $pid; wait $pid 2>/dev/null
}
run reference   --no-fast-path
run fast_host   --no-cuda-graphs --fast-host
run kernels     --no-cuda-graphs --no-fast-host --gdn-kernels fast
run graphs      --cuda-graphs --no-fast-host
run graphs_host --cuda-graphs --fast-host
run fast        --fast-path
run default                        # the folder's own serving defaults; for v3 the same as graphs_host
python - <<'EOF'
import json
for n in ("reference", "fast_host", "kernels", "graphs", "graphs_host", "fast", "default"):
    r = json.load(open(f"latency_{n}.json"))
    l = r["latency"]
    print(f"{n:10} p50 {l['p50_ms']:7.1f}  p95 {l['p95_ms']:7.1f}  mean(p50,p95) {l['mean_of_p50_p95_ms']:7.1f} ms  "
          f"errors {len(r['errors'])}  mean input tokens {r['mean_input_tokens']:.0f}")
EOF
```

Each file has p50, p95, their mean (the speed measure JevBench v1.5 uses), latency by prompt length, and the server's
`/healthz` fields: the fast path in force and where each setting came from (`fast_path_settings`), the kernels, the
weights hash and the warm-up. The requests cycle through
the 83 golden questions, one question per request. The mix includes a few very long prompts that single decisions rarely
have, so read `by_prompt_tokens` too. For another mix, pass a JSON list of request bodies with `--requests`, using
invented data only.

### 6. What to record, and what decides the defaults

- `gpu.txt`, `versions.txt`, `agree_*.json`, `latency_*.json`, `healthz_*.json` and `server_*.log`.
- Peak GPU memory with `--cuda-graphs` (`nvidia-smi` while the server is idle after start-up). The graphs share one
  memory pool, captured largest bucket first.
- A model's defaults are its `manchego_config.json` `serving` field, chosen by whoever publishes the model folder. The
  rule this project applies: `fast_host` must pass `agree` exactly; a configuration that changes the arithmetic is made
  a default only when it passes `agree`, or, as for v3's CUDA graphs, by an explicit decision that is recorded and
  disclosed with the measured shift (above). A configuration that changes a chosen option is not served.

## Failure modes to expect

- **Capture fails at start-up.** Something in the forward is not capturable, for example a host sync inside a kernel
  package. The server does not start, and the log names the error (and, when the graphs came from the model folder,
  the flags that turn them off: `--no-cuda-graphs`). Try `--cuda-graphs` alone (reference kernels) to tell whether the
  graph or the kernels caused it.
- **`kernels` fails `agree` but `graphs` passes.** Check the kernel versions first, and Triton on Hopper.
- **`fast_host` is not exactly 0.** The fast path is then not doing what it claims. Report the question named in
  `max_dprob_question`.


## Validation on Linux (A10, Docker) and the v3 default

Run on 2026-09-30 on a Lambda A10 (Linux, driver 570.148.08) with the Docker image built from commit a10cc27 exactly as the
README says, Manchego v3's bf16 weights (`weights_verified: true`), contract semif, 450 serial single-decision requests over the
invented test questions. Median latency by prompt length, reference path vs CUDA graphs + fast host:

| prompt tokens | reference | graphs + fast host |
|---|---:|---:|
| up to 256 | 54 ms | 71 ms |
| up to 512 | 134 ms | 143 ms |
| up to 1,024 | 190 ms | 275 ms |
| up to 2,048 | 357 ms | 543 ms |

On Linux the padding to a graph size costs more than the captured replay saves, at every length; on Windows (the RTX 5090
above) per-call overhead dominates and the graphs were about five times faster. **Decision (the account holder,
2026-09-30): v3's `manchego_config.json` turns CUDA graphs OFF and keeps the fast host path (bit-for-bit the reference) on.**
`--cuda-graphs` turns the graphs on where they help.
