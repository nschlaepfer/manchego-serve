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
