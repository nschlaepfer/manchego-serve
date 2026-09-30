"""The opt-in fast path (0.2.0.dev, backends/fast_path.py). CUDA is not needed: the kernel switch is tested with fake
kernel modules, CUDA-graph capture with a fake torch.cuda and with an eager stand-in on a tiny random Qwen3.5 model on
CPU. docs/FAST_PATH.md has the checks that need an NVIDIA GPU."""
import json
import types

import pytest

from manchego_serve.backends import fast_path as FP
from manchego_serve.backends import load_backend


# ------------------------------------------------------------------ configuration and command line (no torch)
def test_config_defaults_are_off():
    c = FP.FastPathConfig()
    assert c.is_off and not c.single_path and c.gdn_kernels == "reference" and c.graph_buckets == (256, 512, 1024, 2048)
    assert c.describe() == {"gdn_kernels": "reference", "cuda_graphs": False, "fast_host": False}
    c.check("cpu")


@pytest.mark.parametrize("kw", [{"gdn_kernels": "fla"}, {"graph_buckets": ()}, {"graph_buckets": (256, 256)},
                                {"graph_buckets": (0, 512)}, {"graph_buckets": (256.0,)}, {"graph_buckets": (True,)}])
def test_bad_configs_are_refused(kw):
    with pytest.raises(FP.FastPathError):
        FP.FastPathConfig(**kw)


def test_cuda_only_pieces_are_refused_elsewhere():
    for dev in ("cpu", "mps"):
        FP.FastPathConfig(fast_host=True).check(dev)
        for kw in ({"gdn_kernels": "fast"}, {"cuda_graphs": True}):
            with pytest.raises(FP.FastPathError, match="CUDA"):
                FP.FastPathConfig(**kw).check(dev)
    FP.FastPathConfig(gdn_kernels="fast", cuda_graphs=True, fast_host=True).check("cuda:0")


def test_buckets_are_sorted_and_parsed():
    assert FP.FastPathConfig(graph_buckets=(1024, 256)).graph_buckets == (256, 1024)
    assert FP.parse_buckets("256, 512,2048") == (256, 512, 2048)
    with pytest.raises(FP.FastPathError):
        FP.parse_buckets("256,big")


def test_command_line(monkeypatch):
    from manchego_serve.server import fast_config, parse_args
    for k in ("MANCHEGO_GDN_KERNELS", "MANCHEGO_CUDA_GRAPHS", "MANCHEGO_FAST_HOST", "MANCHEGO_GRAPH_BUCKETS"):
        monkeypatch.delenv(k, raising=False)
    assert fast_config(parse_args([])).is_off
    c = fast_config(parse_args(["--fast-path"]))
    assert (c.gdn_kernels, c.cuda_graphs, c.fast_host) == ("fast", True, True)
    c = fast_config(parse_args(["--cuda-graphs", "--graph-buckets", "512,128"]))
    assert c.cuda_graphs and c.graph_buckets == (128, 512) and c.gdn_kernels == "reference" and not c.fast_host
    assert fast_config(parse_args(["--fast-host"])).describe() == {"gdn_kernels": "reference", "cuda_graphs": False, "fast_host": True}
    monkeypatch.setenv("MANCHEGO_GDN_KERNELS", "fast")
    monkeypatch.setenv("MANCHEGO_CUDA_GRAPHS", "1")
    monkeypatch.setenv("MANCHEGO_FAST_HOST", "true")
    c = fast_config(parse_args([]))
    assert (c.gdn_kernels, c.cuda_graphs, c.fast_host) == ("fast", True, True)
    monkeypatch.setenv("MANCHEGO_CUDA_GRAPHS", "0")
    assert not fast_config(parse_args([])).cuda_graphs


# ------------------------------------------------------------------ serving defaults from manchego_config.json (no torch)
V3_SERVING = {"cuda_graphs": True, "fast_host": True}


def test_settle_nothing_declared_nothing_asked_is_the_reference():
    """The v2.1 case: no `serving` field, no flag. Off, and the report says nothing about model defaults."""
    for backend, device in (("torch", "cuda:0"), ("torch", "cpu"), ("mlx", None)):
        cfg, rep = FP.settle(FP.FastPathRequest(), None, backend, device)
        assert cfg == FP.FastPathConfig() and cfg.is_off
        assert set(rep) == {"in_force", "source"} and set(rep["source"].values()) == {FP.DEFAULT}


def test_settle_applies_the_model_defaults_on_cuda_only():
    cfg, rep = FP.settle(FP.FastPathRequest(), V3_SERVING, "torch", "cuda:0")
    assert (cfg.gdn_kernels, cfg.cuda_graphs, cfg.graph_buckets, cfg.fast_host) == ("reference", True, FP.DEFAULT_BUCKETS, True)
    assert rep["source"] == {"gdn_kernels": FP.DEFAULT, "cuda_graphs": FP.MODEL, "graph_buckets": FP.DEFAULT, "fast_host": FP.MODEL}
    assert rep["model_defaults"] == V3_SERVING and rep["model_defaults_applied"] == ["cuda_graphs", "fast_host"] and "note" not in rep
    for backend, device in (("torch", "cpu"), ("torch", "mps"), ("mlx", None)):
        cfg, rep = FP.settle(FP.FastPathRequest(), V3_SERVING, backend, device)
        assert cfg.is_off and rep["model_defaults_applied"] == [] and "apply only to the torch backend on CUDA" in rep["note"]


def test_settle_flags_override_the_model_defaults():
    cfg, rep = FP.settle(FP.FastPathRequest(cuda_graphs=False), V3_SERVING, "torch", "cuda:1")
    assert not cfg.cuda_graphs and cfg.fast_host and rep["source"]["cuda_graphs"] == FP.ASKED
    cfg, _ = FP.settle(FP.FastPathRequest(fast_host=False), V3_SERVING, "torch", "cuda:0")
    assert cfg.cuda_graphs and not cfg.fast_host
    cfg, rep = FP.settle(FP.FastPathRequest("reference", False, None, False), V3_SERVING, "torch", "cuda:0")
    assert cfg.is_off and rep["model_defaults_applied"] == [] and "note" not in rep
    cfg, _ = FP.settle(FP.FastPathRequest(gdn_kernels="fast"), V3_SERVING, "torch", "cuda:0")
    assert cfg.gdn_kernels == "fast" and cfg.cuda_graphs and cfg.fast_host
    cfg, _ = FP.settle(FP.FastPathRequest(fast_host=True), V3_SERVING, "torch", "cpu")        # asked: honoured anywhere
    assert cfg.fast_host and not cfg.cuda_graphs


def test_settle_graph_buckets():
    serving = {**V3_SERVING, "graph_buckets": [1024, 128]}
    cfg, rep = FP.settle(FP.FastPathRequest(), serving, "torch", "cuda:0")
    assert cfg.graph_buckets == (128, 1024) and rep["source"]["graph_buckets"] == FP.MODEL
    cfg, rep = FP.settle(FP.FastPathRequest(graph_buckets=(512,)), serving, "torch", "cuda:0")
    assert cfg.graph_buckets == (512,) and rep["source"]["graph_buckets"] == FP.ASKED


def test_kernels_are_never_a_model_default():
    from manchego_serve import model_config as MC
    assert "gdn_kernels" not in MC.SERVING_KEYS
    cfg, _ = FP.settle(FP.FastPathRequest(), {"gdn_kernels": "fast", **V3_SERVING}, "torch", "cuda:0")   # settle ignores it too
    assert cfg.gdn_kernels == "reference"


def _config(tmp_path, data):
    from manchego_serve import model_config as MC
    (tmp_path / MC.FILE).write_text(data if isinstance(data, str) else json.dumps(data))
    return MC.read(tmp_path)


def test_model_config_serving(tmp_path):
    none = _config(tmp_path, {"model": "Manchego", "version": "v2.1"})
    assert none.serving is None and "serving" not in none.describe()
    v3 = _config(tmp_path, {"contract": "semif", "serving": V3_SERVING})
    assert v3.serving == V3_SERVING and v3.describe()["serving"] == V3_SERVING
    b = _config(tmp_path, {"serving": {"cuda_graphs": True, "graph_buckets": [256, 2048]}})
    assert b.serving == {"cuda_graphs": True, "graph_buckets": [256, 2048]}
    assert _config(tmp_path, {"serving": {}}).serving == {}


@pytest.mark.parametrize("serving", [True, [], "on", {"cuda_graphs": 1}, {"fast_host": "true"}, {"gdn_kernels": "fast"},
                                     {"cuda_graph": True}, {"graph_buckets": []}, {"graph_buckets": [256, 256]},
                                     {"graph_buckets": [0]}, {"graph_buckets": [256.0]}, {"graph_buckets": [True]},
                                     {"graph_buckets": "256,512"}])
def test_model_config_refuses_a_serving_field_it_cannot_read(tmp_path, serving):
    from manchego_serve import model_config as MC
    with pytest.raises(MC.ModelConfigError, match="serving"):
        _config(tmp_path, {"contract": "semif", "serving": serving})


def test_command_line_switches(monkeypatch):
    from manchego_serve.server import fast_request, parse_args
    for k in ("MANCHEGO_GDN_KERNELS", "MANCHEGO_CUDA_GRAPHS", "MANCHEGO_FAST_HOST", "MANCHEGO_GRAPH_BUCKETS"):
        monkeypatch.delenv(k, raising=False)
    assert fast_request(parse_args([])) == FP.FastPathRequest()                       # nothing said: the model decides
    assert fast_request(parse_args(["--no-cuda-graphs"])) == FP.FastPathRequest(cuda_graphs=False)
    assert fast_request(parse_args(["--no-fast-host"])) == FP.FastPathRequest(fast_host=False)
    assert fast_request(parse_args(["--no-fast-path"])) == FP.FastPathRequest("reference", False, None, False)
    assert fast_request(parse_args(["--fast-path"])) == FP.FastPathRequest("fast", True, None, True)
    assert fast_request(parse_args(["--graph-buckets", "512,128"])).graph_buckets == (512, 128)
    for bad in (["--cuda-graphs", "--no-cuda-graphs"], ["--fast-host", "--no-fast-host"], ["--fast-path", "--no-fast-path"]):
        with pytest.raises(SystemExit):
            parse_args(bad)
    monkeypatch.setenv("MANCHEGO_CUDA_GRAPHS", "0")
    monkeypatch.setenv("MANCHEGO_FAST_HOST", "off")
    assert fast_request(parse_args([])) == FP.FastPathRequest(cuda_graphs=False, fast_host=False)
    assert fast_request(parse_args(["--cuda-graphs"])).cuda_graphs is True                # the flag beats the environment
    monkeypatch.setenv("MANCHEGO_CUDA_GRAPHS", "")
    assert fast_request(parse_args([])).cuda_graphs is None
    monkeypatch.setenv("MANCHEGO_CUDA_GRAPHS", "maybe")
    with pytest.raises(SystemExit):
        parse_args([])


def test_healthz_reports_the_settings_only_when_there_are_some():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from manchego_serve.decider import Decider
    from manchego_serve.server import create_app
    from test_server import FakeBackend
    plain = TestClient(create_app(Decider(FakeBackend(), info={"model_config": {"contract": "auto"}}))).get("/healthz").json()
    assert "fast_path_settings" not in plain and "serving" not in plain["model_config"]
    _, rep = FP.settle(FP.FastPathRequest(), V3_SERVING, "torch", "cpu")
    h = TestClient(create_app(Decider(FakeBackend(), info={"fast_path_settings": rep}))).get("/healthz").json()
    assert h["fast_path_settings"]["note"].startswith("the model folder's serving defaults")


def test_mlx_refuses_the_fast_path():
    with pytest.raises(FP.FastPathError, match="torch backend"):
        load_backend("mlx", "/nonexistent", fast=FP.FastPathConfig(fast_host=True))


# ------------------------------------------------------------------ (a) the kernel switch, with fake kernel modules
def fake_modeling():
    """A stand-in for Transformers' Qwen3.5 modelling module: each name is a wrapper whose __wrapped__ is the reference."""
    calls = []

    def ref(name):
        def reference(query, key, value, g=None, beta=None, chunk_size=64, initial_state=None, output_final_state=False,
                      use_qk_l2norm_in_kernel=False, **kwargs):
            calls.append(("reference", name, sorted(kwargs)))
            return name

        def wrapped(*a, **k):
            return reference(*a, **k)
        wrapped.__wrapped__ = reference
        return wrapped
    mod = types.SimpleNamespace(**{n: ref(n) for n in FP.GDN_FUNCTIONS})
    return mod, calls


def fake_importer(calls, missing=()):
    def fast(name):
        def kernel(q, k, v, g=None, beta=None, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None,
                   use_qk_l2norm_in_kernel=False):
            calls.append(("fast", name))
            return "fast:" + name
        return kernel

    def importer(module):
        if module.split(".")[0] in missing:
            raise ImportError(f"No module named {module!r}")
        return types.SimpleNamespace(**{func: fast(func) for _, m, func in FP.GDN_FUNCTIONS.values() if m == module})
    return importer


def test_fast_kernels_where_importable():
    mod, calls = fake_modeling()
    report = FP.select_gdn_kernels("fast", mod, fake_importer(calls))
    assert report == {n: f"{m}.{f}" for n, (_, m, f) in FP.GDN_FUNCTIONS.items()}
    # the layers' extra keyword arguments are dropped, as Transformers' own wrapper drops them
    assert mod.torch_chunk_gated_delta_rule(1, 2, 3, g=0, beta=0, chunk_size=64, use_qk_l2norm_in_kernel=True, cu_seqlens=None,
                                            output_attentions=False) == "fast:chunk_gated_delta_rule"
    assert mod.causal_conv1d_fn(1, 2, 3) == "fast:causal_conv1d_fn"


def test_fast_kernels_fall_back_one_by_one():
    mod, calls = fake_modeling()
    report = FP.select_gdn_kernels("fast", mod, fake_importer(calls, missing=("causal_conv1d",)))
    assert report["torch_chunk_gated_delta_rule"] == "fla.ops.gated_delta_rule.chunk_gated_delta_rule"
    assert report["causal_conv1d_fn"].startswith("reference (causal-conv1d not importable: ImportError")
    assert mod.causal_conv1d_fn(1, 2, 3) == "causal_conv1d_fn" and calls[-1][:2] == ("reference", "causal_conv1d_fn")


def test_reference_restores_after_fast(monkeypatch):
    mod, calls = fake_modeling()
    FP.select_gdn_kernels("fast", mod, fake_importer(calls))
    report = FP.select_gdn_kernels("reference", mod)
    assert set(report.values()) == {"reference"}
    for n in FP.GDN_FUNCTIONS:
        assert getattr(mod, n)(1, 2, 3, foo=1) == n              # the reference again, extra kwargs filtered by name
    assert all(c[2] == [] for c in calls if c[0] == "reference")


def test_reference_is_forced_when_the_kernels_are_installed(monkeypatch):
    """0.1.x used fla / causal-conv1d whenever Transformers could import them; 0.2.0's default forces the reference."""
    mod, calls = fake_modeling()
    untouched = dict(vars(mod))
    report = FP.select_gdn_kernels("reference", mod)
    assert vars(mod) == untouched and all("not installed" in v for v in report.values())
    monkeypatch.setattr(FP, "_importable", lambda module: True)
    report = FP.select_gdn_kernels("reference", mod)
    assert all(v.startswith("reference (forced;") for v in report.values())
    assert mod.torch_chunk_gated_delta_rule(1, 2, 3) == "torch_chunk_gated_delta_rule"


def test_absent_names_are_reported():
    mod = types.SimpleNamespace()
    assert set(FP.select_gdn_kernels("reference", mod).values()) == {"absent from this Transformers version"}


# ------------------------------------------------------------------ (b) CUDA-graph capture, with a fake torch.cuda
def test_cuda_capture_uses_the_graph_api():
    log = []

    class Ctx:
        def __init__(self, what):
            self.what = what

        def __enter__(self):
            log.append(("enter", self.what))

        def __exit__(self, *a):
            log.append(("exit", self.what))

    class Stream:
        def wait_stream(self, other):
            log.append("wait_stream")

    class Graph:
        def replay(self):
            log.append("replay")

    cuda = types.SimpleNamespace(graph_pool_handle=lambda: "pool", Stream=Stream, current_stream=Stream,
                                 stream=lambda s: Ctx("side-stream"), CUDAGraph=Graph,
                                 graph=lambda g, pool: Ctx(("capture", pool)), synchronize=lambda: log.append("sync"))
    fake_torch = types.SimpleNamespace(cuda=cuda)
    steps = []
    replay, out, pool = FP.cuda_capture(lambda: steps.append(len(log)) or "OUT", None, 3, fake_torch)
    assert out == "OUT" and pool == "pool" and len(steps) == 4                         # 3 warm-up runs, 1 captured
    assert log.index(("enter", "side-stream")) < log.index(("exit", "side-stream")) < log.index(("enter", ("capture", "pool")))
    assert log[-1] == "sync"
    replay()
    assert log[-1] == "replay"
    _, _, pool2 = FP.cuda_capture(lambda: "OUT", "shared", 0, fake_torch)
    assert pool2 == "shared"


# ------------------------------------------------------------------ a tiny random Qwen3.5 on CPU
@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=1024, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
                            max_position_embeddings=4096, tie_word_embeddings=True, pad_token_id=0)
    return torch, Qwen3_5ForCausalLM(cfg).eval()


class Tok:
    """Enough tokenizer for the backend and the Decider (test_server.FakeTok) plus a pad id."""
    pad_token_id = 0

    def __init__(self):
        from test_server import FakeTok
        self._t = FakeTok()

    def encode(self, *a, **k):
        return self._t.encode(*a, **k)

    def apply_chat_template(self, *a, **k):
        return self._t.apply_chat_template(*a, **k)


def prompts(torch, lengths=(3, 17, 64, 65, 200, 256, 300)):
    g = torch.Generator().manual_seed(1)
    return [(torch.randint(1, 1000, (n,), generator=g).tolist(), torch.randint(1, 1000, (k,), generator=g).tolist())
            for n, k in zip(lengths, (2, 3, 16, 5, 26, 2, 40))]


def backend(tiny, **fast):
    from manchego_serve.backends.torch_backend import TorchBackend
    return TorchBackend.from_model(tiny[1], Tok(), device="cpu", fast=FP.FastPathConfig(**fast))


def test_default_backend_selects_nothing(tiny):
    b = backend(tiny)
    assert b.fast_path is None and b._single is None
    assert all(v.startswith("reference") for v in b.runtime["gdn_kernels"].values())


def test_fast_host_is_bit_for_bit_the_reference(tiny):
    torch, _ = tiny
    ref, fast = backend(tiny), backend(tiny, fast_host=True)
    assert fast.fast_path == {"gdn_kernels": "reference", "cuda_graphs": False, "fast_host": True,
                              "gdn_kernels_in_use": fast.runtime["gdn_kernels"], "graphs": False}
    for ids, cands in prompts(torch):
        assert fast.logits_batch([(ids, cands)]) == ref.logits_batch([(ids, cands)])
    assert fast._single.counts == {"graph": 0, "eager": len(prompts(torch))}
    fast.logits_batch([(ids, cands)])
    assert len(fast._single.rows) == len({tuple(c) for _, c in prompts(torch)})


def test_fast_host_is_bit_for_bit_on_mps(tiny):
    """Regression: a non-blocking copy from pageable host memory to MPS reads freed memory; the fast path copies
    non-blocking only from pinned memory (CUDA)."""
    torch, m = tiny
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    from manchego_serve.backends.torch_backend import TorchBackend
    m_mps = type(m)(m.config).eval()
    m_mps.load_state_dict(m.state_dict())
    m_mps.to("mps")
    ref = TorchBackend.from_model(m_mps, Tok(), device="mps")
    fast = TorchBackend.from_model(m_mps, Tok(), device="mps", fast=FP.FastPathConfig(fast_host=True))
    for _ in range(3):
        for ids, cands in prompts(torch):
            assert fast.logits_batch([(ids, cands)]) == ref.logits_batch([(ids, cands)])


def test_the_no_mask_mapping_is_the_all_ones_mask(tiny):
    torch, m = tiny
    ids = torch.tensor([prompts(torch)[3][0]])
    with torch.inference_mode():
        a = m.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        b = m.model(input_ids=ids, attention_mask=FP.no_masks(m), use_cache=False).last_hidden_state
    assert torch.equal(a, b)


def test_graph_buckets_with_an_eager_stand_in(tiny):
    torch, m = tiny
    gb = FP.GraphBuckets(torch, m, "cpu", (64, 128, 256), pad_id=0, capture=FP.eager_capture, pin=False)
    assert gb.capture_all() == {"64": "captured", "128": "captured", "256": "captured"}
    assert [gb.bucket_for(n) for n in (1, 64, 65, 256, 257)] == [64, 64, 128, 256, None]
    for ids, _ in prompts(torch):
        h = gb.hidden_last(ids)
        L = gb.bucket_for(len(ids))
        if L is None:
            assert h is None
            continue
        padded = torch.zeros((1, L), dtype=torch.long)
        padded[0, : len(ids)] = torch.tensor(ids)
        with torch.inference_mode():
            same_shape = m.model(input_ids=padded, attention_mask=FP.no_masks(m), use_cache=False).last_hidden_state[0, len(ids) - 1]
            exact = m.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0, -1]
        assert torch.equal(h, same_shape.float())                     # the replay is the padded forward
        assert torch.allclose(h, exact.float(), atol=1e-5, rtol=0)    # padding after the read position cannot reach it
    # the static buffers are reused: an earlier, longer prompt leaves nothing behind
    a = gb.hidden_last(list(range(1, 60))).clone()
    gb.hidden_last(list(range(1, 64)))
    assert torch.equal(gb.hidden_last(list(range(1, 60))), a)


def test_backend_with_graphs_matches_the_reference_readout(tiny, monkeypatch):
    """cuda_graphs on CPU is refused; with the capture swapped for the eager stand-in, the backend's graph path is the
    padded forward read the reference way."""
    torch, m = tiny
    monkeypatch.setattr(FP.FastPathConfig, "check", lambda self, device: None)
    monkeypatch.setattr(FP, "cuda_capture", FP.eager_capture)
    import manchego_serve.backends.torch_backend as TB
    monkeypatch.setattr(TB, "GraphBuckets", lambda *a, **k: FP.GraphBuckets(*a, **{**k, "capture": FP.eager_capture}))
    ref, fast = backend(tiny), backend(tiny, cuda_graphs=True, graph_buckets=(64, 256), fast_host=True)
    assert fast.fast_path["graphs"] == {"64": "captured", "256": "captured"}
    worst = 0.0
    for ids, cands in prompts(torch):
        z_ref, z_fast = ref.logits_batch([(ids, cands)])[0], fast.logits_batch([(ids, cands)])[0]
        worst = max(worst, max(abs(a - b) for a, b in zip(z_ref, z_fast)))
        assert z_ref.index(max(z_ref)) == z_fast.index(max(z_fast))
    assert worst < 1e-4
    assert fast._single.counts == {"graph": 6, "eager": 1}            # the 300-token prompt is longer than every bucket


def test_batches_always_run_the_reference_code(tiny):
    torch, _ = tiny
    ref, fast = backend(tiny), backend(tiny, fast_host=True)
    items = prompts(torch)[:3]
    assert fast.logits_batch(items) == ref.logits_batch(items)
    assert fast._single.counts == {"graph": 0, "eager": 0}


def test_cuda_pieces_refused_on_cpu(tiny):
    for kw in ({"gdn_kernels": "fast"}, {"cuda_graphs": True}):
        with pytest.raises(FP.FastPathError):
            backend(tiny, **kw)


def test_kernel_switch_reaches_the_layers(tiny):
    """The Qwen3.5 layers look the kernel names up at call time: a (fake) fast kernel selected after loading is the one
    that runs, and selecting the reference again gives the reference numbers bit for bit."""
    torch, m = tiny
    from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ
    ids = torch.tensor([prompts(torch)[2][0]])
    with torch.inference_mode():
        before = m.model(input_ids=ids, use_cache=False).last_hidden_state
    seen = []
    ref_chunk, ref_conv = FP._reference_of(MQ.torch_chunk_gated_delta_rule), FP._reference_of(MQ.causal_conv1d_fn)

    def importer(module):
        def chunk_gated_delta_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                                   use_qk_l2norm_in_kernel=False):
            seen.append("chunk")
            return ref_chunk(query, key, value, g, beta, initial_state=initial_state, output_final_state=output_final_state,
                             use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

        def causal_conv1d_fn(x, weight, bias=None, activation=None):
            seen.append("conv")
            return ref_conv(x, weight, bias, activation=activation)
        return types.SimpleNamespace(chunk_gated_delta_rule=chunk_gated_delta_rule, fused_recurrent_gated_delta_rule=None,
                                     causal_conv1d_fn=causal_conv1d_fn, causal_conv1d_update=None)
    try:
        FP.select_gdn_kernels("fast", MQ, importer)
        with torch.inference_mode():
            during = m.model(input_ids=ids, use_cache=False).last_hidden_state
        assert seen.count("chunk") == 3 and seen.count("conv") == 3   # three linear-attention layers
        assert torch.equal(before, during)                             # the fakes delegate to the reference
    finally:
        FP.select_gdn_kernels("reference", MQ)
    assert all(getattr(MQ, n).manchego_origin == "reference" for n in FP.GDN_FUNCTIONS)
    with torch.inference_mode():
        after = m.model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(before, after)


def test_decider_end_to_end_on_the_fast_host_path(tiny):
    from manchego_serve.decider import Decider
    from test_server import CHOICE, NOUL, SCORE, STATE, menu
    body = {"state": STATE, "questions": {"c": CHOICE, "n": NOUL, "s": SCORE, "m": {"type": "choice", "instructions": "x", "criteria": menu(40)}}}
    for contract in ("auto", "semif"):
        ref = Decider(backend(tiny), contract=contract).handle(body)
        fast = Decider(backend(tiny, fast_host=True), contract=contract).handle(body)
        assert json.dumps(fast["answers"]) == json.dumps(ref["answers"])
        assert "fast_path" not in ref["manchego"] and fast["manchego"]["fast_path"]["fast_host"] is True
