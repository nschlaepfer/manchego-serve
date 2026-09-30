"""server.build() end to end on a tiny random Qwen3.5 saved to a folder with the real tokenizer (needs torch and
MANCHEGO_TEST_TOKENIZER): the folder's manchego_config.json picks the contract and the served name, the default
temperature map stays off for weights it does not list, a schema-2 map binds by hash or stops the start-up, and the
fast-path flags reach the backend."""
import json
import os
import shutil

import pytest

from manchego_serve import temperature as TM

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")


@pytest.fixture(scope="module")
def tiny_folder(tmp_path_factory):
    torch = pytest.importorskip("torch")
    tok = os.environ.get("MANCHEGO_TEST_TOKENIZER")
    if not tok:
        pytest.skip("set MANCHEGO_TEST_TOKENIZER to a Manchego / Qwen3.5 folder (its tokenizer files are copied)")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=248320, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                            linear_key_head_dim=16, linear_value_head_dim=16, full_attention_interval=4, tie_word_embeddings=True)
    d = tmp_path_factory.mktemp("tiny-qwen35")
    Qwen3_5ForCausalLM(cfg).save_pretrained(d)
    for n in TOKENIZER_FILES:
        if os.path.isfile(os.path.join(tok, n)):
            shutil.copy(os.path.join(tok, n), d / n)
    return d


def build(folder, logs, **kw):
    from manchego_serve.server import build as server_build
    return server_build("torch", str(folder), None, dtype="float32", device="cpu", log=logs.append, **kw)


BODY = {"state": "Order 7: the kettle arrived with a cracked lid.",
        "questions": {"n": {"type": "noul", "instructions": "Is the item damaged?"},
                      "c": {"type": "choice", "instructions": "Which team?", "criteria": {f"t{i}": None for i in range(20)}}}}


def test_build_follows_the_model_config(tiny_folder):
    from manchego_serve.backends.fast_path import FastPathConfig
    (tiny_folder / "manchego_config.json").write_text(json.dumps({"model": "Tiny", "version": "t1", "contract": "semif",
                                                                   "served_name": "tiny-1"}))
    logs = []
    d = build(tiny_folder, logs, fast=FastPathConfig(fast_host=True))
    assert d.contract == "semif" and d.model_name == "tiny-1" and d.info["model_label"] == "Tiny t1"
    assert d.info["model_config"]["contract_source"] == "manchego_config.json"
    assert d.tmap.is_identity and "contract auto" in d.tmap.note            # the v2.1 map: contract auto only (and not these weights)
    assert d.b.fast_path["fast_host"] is True and d.b.runtime["gdn_kernels"]
    out = d.handle(BODY)
    assert out["model"] == "tiny-1" and out["manchego"]["contract_by_question"] == {"n": "semif", "c": "state_first"}
    assert out["manchego"]["temperature"] == 1.0 and out["manchego"]["fast_path"]["fast_host"] is True
    assert any(m.startswith("contract: semif (manchego_config.json)") for m in logs)


def test_build_contract_override_and_default(tiny_folder):
    (tiny_folder / "manchego_config.json").write_text(json.dumps({"model": "Tiny"}))
    d = build(tiny_folder, [])
    assert d.contract == "auto" and d.model_name == "manchego-2.1" and d.b.fast_path is None
    assert d.tmap.is_identity and "not one of them" in d.tmap.note          # the v2.1 map does not list these weights
    assert d.handle(BODY)["manchego"]["contract_by_question"] == {"n": "short", "c": "short"}
    d = build(tiny_folder, [], contract="semif")
    assert d.contract == "semif" and d.info["model_config"]["contract_source"].startswith("--contract")


def test_build_binds_a_schema_2_map_by_hash(tiny_folder, tmp_path):
    from manchego_serve.weights import identify
    (tiny_folder / "manchego_config.json").write_text(json.dumps({"contract": "semif"}))
    sha = identify(str(tiny_folder))["weights_sha256"]
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"schema": TM.SCHEMA_V2, "temperatures": {"choice": 0.7, "noul": 0.5, "score": 1.0}, "model_sha256": sha,
                                "fitted_on": "test", "rule": "test", "contract": "semif"}))
    d = build(tiny_folder, [], temperature_map=str(good))
    assert d.tmap.temperatures == {"choice": 0.7, "noul": 0.5, "score": 1.0} and d.tmap.binding.startswith("bound")
    assert d.handle(BODY)["manchego"]["temperature_by_question"] == {"n": 0.5, "c": 0.7}
    bad = tmp_path / "bad.json"
    bad.write_text(good.read_text().replace(sha, "0" * 64))
    with pytest.raises(TM.TemperatureMapError, match="refusing"):
        build(tiny_folder, [], temperature_map=str(bad))
    with pytest.raises(TM.TemperatureMapError, match="contract"):
        build(tiny_folder, [], temperature_map=str(good), contract="auto")
    with pytest.raises(TM.TemperatureMapError, match="not hashed"):
        build(tiny_folder, [], temperature_map=str(good), hash_weights=False)


def test_build_refuses_a_bad_model_config(tiny_folder):
    from manchego_serve.model_config import ModelConfigError
    (tiny_folder / "manchego_config.json").write_text(json.dumps({"contract": "short"}))
    with pytest.raises(ModelConfigError):
        build(tiny_folder, [])
    (tiny_folder / "manchego_config.json").unlink()


# ------------------------------------------------------------------ serving defaults (0.2.0)
V3_CONFIG = {"model": "Tiny", "version": "t3", "contract": "semif", "served_name": "tiny-3",
             "serving": {"cuda_graphs": True, "fast_host": True, "graph_buckets": [64, 256]}}


def test_build_without_serving_reports_nothing_new(tiny_folder):
    """A folder with no `serving` field (the published v2.1 folder) and no flag: no fast path and no new report."""
    (tiny_folder / "manchego_config.json").write_text(json.dumps({"model": "Tiny"}))
    d = build(tiny_folder, [])
    assert d.b.fast_path is None and "fast_path_settings" not in d.info and "serving" not in d.info["model_config"]
    assert "fast_path" not in d.handle(BODY)["manchego"]


def test_build_serving_defaults_are_not_applied_off_cuda(tiny_folder):
    (tiny_folder / "manchego_config.json").write_text(json.dumps(V3_CONFIG))
    logs = []
    d = build(tiny_folder, logs)                                          # device cpu: silently off, with a note
    s = d.info["fast_path_settings"]
    assert d.b.fast_path is None and s["model_defaults_applied"] == [] and "not applied" in s["note"]
    assert d.info["model_config"]["serving"] == V3_CONFIG["serving"]
    assert any("not applied" in m for m in logs)
    from manchego_serve.backends.fast_path import FastPathRequest
    d = build(tiny_folder, [], fast=FastPathRequest(fast_host=True))     # asked explicitly: honoured on the CPU
    assert d.b.fast_path["fast_host"] is True and d.info["fast_path_settings"]["source"]["fast_host"].startswith("command line")


def _as_if_cuda(monkeypatch):
    """settle() sees a CUDA device and the graphs are captured with the eager stand-in (the model still runs on the CPU)."""
    from manchego_serve.backends import fast_path as FP
    import manchego_serve.backends.torch_backend as TB
    real = FP.settle
    monkeypatch.setattr(FP, "settle", lambda request, serving, backend, device: real(request, serving, backend, "cuda:0"))
    monkeypatch.setattr(FP.FastPathConfig, "check", lambda self, device: None)
    monkeypatch.setattr(TB, "GraphBuckets", lambda *a, **k: FP.GraphBuckets(*a, **{**k, "capture": FP.eager_capture, "pin": False}))
    monkeypatch.setattr(TB.TorchBackend, "_setup", _unpinned(TB.TorchBackend._setup))


def _unpinned(setup):
    def wrapped(self, *a, **k):
        setup(self, *a, **k)
        if self._single is not None:
            self._single.pin = False                                          # no pinned memory without CUDA
    return wrapped


def test_build_applies_serving_defaults_on_cuda(tiny_folder, monkeypatch):
    from manchego_serve.backends.fast_path import MODEL, FastPathRequest
    (tiny_folder / "manchego_config.json").write_text(json.dumps(V3_CONFIG))
    _as_if_cuda(monkeypatch)
    d = build(tiny_folder, [])
    s = d.info["fast_path_settings"]
    assert s["in_force"] == {"gdn_kernels": "reference", "cuda_graphs": [64, 256], "fast_host": True} and "note" not in s
    assert s["source"]["cuda_graphs"] == s["source"]["graph_buckets"] == s["source"]["fast_host"] == MODEL
    assert d.b.fast_path["graphs"] == {"64": "captured", "256": "captured"}
    out = d.handle(BODY)
    assert out["manchego"]["fast_path"]["cuda_graphs"] == [64, 256] and d.b._single.counts["graph"] == 2
    d = build(tiny_folder, [], fast=FastPathRequest(cuda_graphs=False))  # --no-cuda-graphs
    assert d.b.fast_path["cuda_graphs"] is False and d.b.fast_path["fast_host"] is True
    d = build(tiny_folder, [], fast=FastPathRequest("reference", False, None, False))     # --no-fast-path
    assert d.b.fast_path is None and d.info["fast_path_settings"]["model_defaults_applied"] == []


def test_build_names_the_serving_field_when_the_fast_path_fails(tiny_folder, monkeypatch):
    from manchego_serve.backends import fast_path as FP
    (tiny_folder / "manchego_config.json").write_text(json.dumps(V3_CONFIG))
    real = FP.settle
    monkeypatch.setattr(FP, "settle", lambda request, serving, backend, device: real(request, serving, backend, "cuda:0"))
    logs = []
    with pytest.raises(FP.FastPathError, match="CUDA"):                  # the real check: this backend runs on the CPU
        build(tiny_folder, logs)
    assert any("--no-cuda-graphs" in m for m in logs)
    (tiny_folder / "manchego_config.json").unlink()


def test_build_applies_the_packaged_v3_map_by_hash_only(tiny_folder, monkeypatch):
    """The default map for v3 reaches the server by weights_sha256 (the tiny model's hash stands in for a v3 build) and
    contract semif; off, another contract or unhashed weights give T = 1.0."""
    from manchego_serve.weights import identify
    (tiny_folder / "manchego_config.json").write_text(json.dumps({"contract": "semif"}))
    assert build(tiny_folder, []).tmap.is_identity                           # not a v3 build
    sha = identify(str(tiny_folder))["weights_sha256"]
    monkeypatch.setitem(TM.V3_BUILDS, sha, {"build": "tiny (standing in for a v3 MLX build)", "note": TM.V3_MLX_NOTE})
    d = build(tiny_folder, [])
    assert d.tmap.temperatures == {"choice": 1.5, "noul": 0.2, "score": 1.0} and TM.V3_MLX_NOTE in d.tmap.binding
    assert d.handle(BODY)["manchego"]["temperature_by_question"] == {"n": 0.2, "c": 1.5}
    assert build(tiny_folder, [], temperature_map="off").tmap.is_identity
    assert "contract semif" in build(tiny_folder, [], contract="auto").tmap.note
    assert "not hashed" in build(tiny_folder, [], hash_weights=False).tmap.note
    (tiny_folder / "manchego_config.json").unlink()


def test_build_reads_a_version_name_as_the_pinned_commit(tiny_folder):
    """The Docker image declares its folder with MANCHEGO_REVISION=v2.1 or v3 (a version name for the pinned commit)."""
    from manchego_serve.server import build as server_build
    from manchego_serve.weights import PINS, RevisionError
    d = server_build("torch", str(tiny_folder), "v2.1", dtype="float32", device="cpu", log=lambda m: None)
    assert d.info["revision"] == PINS["oraculumai/Manchego"]["revision"] and d.info["weights_verified"] is False
    from manchego_serve.weights import V3_REVISION
    d3 = server_build("torch", str(tiny_folder), "v3", dtype="float32", device="cpu", log=lambda m: None)
    assert d3.info["revision"] == V3_REVISION and d3.info["weights_verified"] is False           # tiny weights match no pin
