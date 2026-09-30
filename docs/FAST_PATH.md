# The fast path (0.2.0.dev): what it is and how to validate it on an NVIDIA GPU

Every piece is **off by default**. With all of them off, the torch backend runs the 0.1.x code (the published numbers'
arithmetic). None of this has run on CUDA yet: the code was written and tested on a Mac without CUDA (fake kernel
modules, a fake `torch.cuda`, an eager stand-in for graph capture, and a tiny random Qwen3.5 on CPU). This page lists
the exact commands that validate it on an NVIDIA machine, and what to record.

| flag | what it does | arithmetic | device |
|---|---|---|---|
| `--gdn-kernels reference` (default) | Qwen3.5's gated-delta-net (linear-attention) layers run Transformers' reference PyTorch functions, **even when** flash-linear-attention or causal-conv1d is installed. (0.1.x used those packages silently whenever Transformers could import them.) | the published path | any |
| `--gdn-kernels fast` | fla's `chunk_gated_delta_rule` (and `fused_recurrent_gated_delta_rule`) and causal-conv1d's `causal_conv1d_fn` (and `causal_conv1d_update`), each where importable. `/healthz` → `runtime.gdn_kernels` names what runs for each function. | changes (bf16 kernel arithmetic) | CUDA |
| `--cuda-graphs` (`--graph-buckets 256,512,1024,2048`) | Each one-prompt forward is replayed as a CUDA graph captured at start-up, at the smallest padded length that holds the prompt. There is no attention mask and no KV cache, and the hidden state is read at the prompt's own last position. Longer prompts and padded batches run eagerly. | changes slightly: the padding cannot reach the answer (every layer is causal, the padding comes after the read position), but the kernels see another shape | CUDA |
| `--fast-host` | The eager one-prompt path without host work it does not need. Ids and option-code ids go to the device in one pinned non-blocking copy each, and there is no all-ones attention mask (Transformers checks one on the host twice per forward, then drops it). Option-code rows of the output matrix are cached per code set, and there is one device-to-host copy per question. | **bit for bit** the reference | any |
| `--fast-path` | all three | changes | CUDA |

Environment variables: `MANCHEGO_GDN_KERNELS=fast`, `MANCHEGO_CUDA_GRAPHS=1`, `MANCHEGO_GRAPH_BUCKETS=...` and
`MANCHEGO_FAST_HOST=1`. Responses carry `manchego.fast_path` only when a piece is on.

**The mask.** The text model accepts a precomputed `{layer type: mask}` mapping. The fast path passes `{type: None}`.
That is exactly what Transformers builds from an all-ones mask or from no mask: SDPA with `is_causal=True`, and no
padding mask in the recurrent layers. It matters for the graphs too. Transformers' `is_tracing()` is true during CUDA
stream capture, and inside a capture its mask builder would switch SDPA to an explicit 4D mask, which is a different
kernel from the eager path.

## What was measured here (Apple M3 Max; no CUDA)

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
  - Padding alone uses more than half of the 0.03 budget on one question. On CUDA, graphs and kernels together must
    still fit inside it.
  - An earlier run of the same preview found `fast_host` far off: max |Δp| 0.96 and 48 changed choices. The cause was a
    non-blocking copy from a temporary pageable CPU tensor, which on MPS reads freed memory every time. It is fixed (the
    fast path copies non-blocking only from pinned memory, that is on CUDA) and covered by a test that runs on MPS.

## Validation on an NVIDIA machine

Record every output file, `nvidia-smi`, and the package versions printed in step 2.

### 1. Environment

```bash
git clone <this repository> manchego-serve && cd manchego-serve && git checkout fast-path
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
{"model": "Manchego", "version": "v3", "contract": "semif", "served_name": "manchego-3"}
```

The published v2.1 folder has no `contract` field, which means `auto`. Check both models if both will be served. The
server never downloads anything (`manchego-serve-download` fetches the published v2.1 once).

### 4. Numerical agreement against the reference path (golden requests, one loaded model)

```bash
python tools/fast_path_check.py agree --model /models/manchego-v3 --out agree_v3.json
python tools/fast_path_check.py agree --model /models/manchego    --out agree_v21.json     # v2.1, contract auto
```

The tool loads the model once in bf16 and scores all 83 golden questions with the reference path, one question per
forward pass, under the model's contract. The questions are the 46 of `tests/fixtures/requests.json` and the 37 of
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
run reference
run fast_host   --fast-host
run kernels     --gdn-kernels fast
run graphs      --cuda-graphs
run fast        --fast-path
python - <<'EOF'
import json
for n in ("reference", "fast_host", "kernels", "graphs", "fast"):
    r = json.load(open(f"latency_{n}.json"))
    l = r["latency"]
    print(f"{n:10} p50 {l['p50_ms']:7.1f}  p95 {l['p95_ms']:7.1f}  mean(p50,p95) {l['mean_of_p50_p95_ms']:7.1f} ms  "
          f"errors {len(r['errors'])}  mean input tokens {r['mean_input_tokens']:.0f}")
EOF
```

Each file has p50, p95, their mean (the speed measure JevBench v1.5 uses), latency by prompt length, and the server's
`/healthz` fields: the fast path in force, the kernels, the weights hash and the warm-up. The requests cycle through
the 83 golden questions, one question per request. The mix includes a few very long prompts that single decisions rarely
have, so read `by_prompt_tokens` too. For another mix, pass a JSON list of request bodies with `--requests`, using
invented data only.

### 6. What to record, and what decides the defaults

- `gpu.txt`, `versions.txt`, `agree_*.json`, `latency_*.json`, `healthz_*.json` and `server_*.log`.
- Peak GPU memory with `--cuda-graphs` (`nvidia-smi` while the server is idle after start-up). The graphs share one
  memory pool, captured largest bucket first.
- The flags stay opt-in until these results are recorded and reviewed. A configuration that fails `agree` is not served,
  whatever it gains in speed.

## Failure modes to expect

- **Capture fails at start-up.** Something in the forward is not capturable, for example a host sync inside a kernel
  package. The server does not start, and the log names the error. Try `--cuda-graphs` alone (reference kernels) to
  tell whether the graph or the kernels caused it.
- **`kernels` fails `agree` but `graphs` passes.** Check the kernel versions first, and Triton on Hopper.
- **`fast_host` is not exactly 0.** The fast path is then not doing what it claims. Report the question named in
  `max_dprob_question`.
