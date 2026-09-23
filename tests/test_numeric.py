"""Numeric parity. Needs weights, so each test skips unless its weights are available.

  MLX (Apple silicon): this package's MLX backend on oraculumai/Manchego-MLX-8bit@79e55e2d must reproduce the reference
  implementation's answers (golden_mlx8bit.json) to 1e-6, in both executions. Set MANCHEGO_TEST_MLX_MODEL to the folder,
  or have the pinned snapshot in the local Hugging Face cache.

  torch (informational): the torch backend on oraculumai/Manchego@77403228 (bf16 weights, run in float32 on CPU, or
  bf16 on CUDA) against the MLX 8-bit reference on the requests marked `torch`. Different precision, so the tolerance
  is loose; the maximum difference is printed (pytest -s). Set MANCHEGO_TEST_TORCH_MODEL.
"""
import json
import math

import pytest

from conftest import BY_ID, REQUESTS, local_or_cached, load
from manchego_serve.weights import PINS

GOLDEN = load("golden_mlx8bit.json")
TOL = 1e-6


def numbers(answer: dict) -> dict:
    """Every number of one wire answer, keyed by where it sits."""
    out = {}
    for k in ("noul", "score", "confidence"):
        if k in answer:
            out[k] = answer[k]
    for v, p in answer.get("probabilities", {}).items():
        out[f"p[{v}]"] = p
    return out


def compare(ours: dict, ref: dict) -> float:
    """Structure must be identical; returns the largest absolute difference over every number."""
    assert ours.keys() == ref.keys()
    worst = 0.0
    for name in ref:
        a, b = ours[name], ref[name]
        assert a["type"] == b["type"]
        assert a.get("choice") == b.get("choice")
        assert a.get("legend") == b.get("legend")
        assert list(a.get("probabilities", {})) == list(b.get("probabilities", {}))
        na, nb = numbers(a), numbers(b)
        assert na.keys() == nb.keys()
        for k in nb:
            assert math.isfinite(na[k])
            worst = max(worst, abs(na[k] - nb[k]))
        if "probabilities" in a:
            assert abs(sum(a["probabilities"].values()) - 1.0) < 1e-9
    return worst


@pytest.fixture(scope="module")
def mlx_backend():
    pytest.importorskip("mlx_lm")
    pin = PINS["oraculumai/Manchego-MLX-8bit"]
    path = local_or_cached("MANCHEGO_TEST_MLX_MODEL", "oraculumai/Manchego-MLX-8bit", pin["revision"])
    if not path:
        pytest.skip("set MANCHEGO_TEST_MLX_MODEL to the Manchego-MLX-8bit v2.1 folder")
    from manchego_serve.backends.mlx_backend import MLXBackend
    return MLXBackend(path)


@pytest.mark.parametrize("execution", ["sequential", "batched"])
def test_mlx_matches_reference(mlx_backend, execution):
    from manchego_serve.decider import Decider
    dec = Decider(mlx_backend, execution=execution)
    ref = GOLDEN["results"][execution]
    assert len(ref) == 20
    worst = 0.0
    for rid, r in ref.items():
        out = dec.handle(BY_ID[rid]["body"])
        worst = max(worst, compare(out["answers"], r["answers"]))
        # 0.1.2: the Decider's default temperature map is `off`, which must be the v0.1.0 readout byte for byte
        assert json.dumps(out["answers"]) == json.dumps(r["answers"]), rid
        assert out["usage"]["input_tokens"] == r["input_tokens"]
        assert out["manchego"]["contract_by_question"] == r["contract_by_question"]
        assert out["manchego"]["forward_passes"] == r["backbone_calls"]
    print(f"\nMLX 8-bit, {execution}: {len(ref)} requests, max |difference| vs reference = {worst:.3g}")
    assert worst <= TOL


def test_mlx_default_map_keeps_every_choice(mlx_backend):
    """0.1.2: the fitted map on the published MLX 8-bit build changes probabilities, never a chosen option."""
    from manchego_serve import temperature as TM
    from manchego_serve.decider import Decider
    tmap = TM.load("default")
    assert not tmap.is_identity
    dec = Decider(mlx_backend, temperature_map=tmap)
    for rid, r in GOLDEN["results"]["sequential"].items():
        out = dec.handle(BY_ID[rid]["body"])
        for name, a in out["answers"].items():
            g = r["answers"][name]
            assert a.get("choice") == g.get("choice")
            if "probabilities" in a:
                pa, pg = list(a["probabilities"].values()), list(g["probabilities"].values())
                assert pa.index(max(pa)) == pg.index(max(pg))
            else:
                assert (a["noul"] > 0.5) == (g["noul"] > 0.5)


def test_torch_vs_mlx_reference_informational():
    pytest.importorskip("torch")
    pin = PINS["oraculumai/Manchego"]
    path = local_or_cached("MANCHEGO_TEST_TORCH_MODEL", "oraculumai/Manchego", pin["revision"])
    if not path:
        pytest.skip("set MANCHEGO_TEST_TORCH_MODEL to the Manchego v2.1 bf16 folder")
    import os
    from manchego_serve.backends.torch_backend import TorchBackend
    from manchego_serve.decider import Decider
    b = TorchBackend(path, dtype=os.environ.get("MANCHEGO_TEST_TORCH_DTYPE", "auto"))
    dec = Decider(b, execution="sequential")
    chosen = [r for r in REQUESTS if r["torch"]]
    assert len(chosen) == 5
    worst, flips = 0.0, 0
    for r in chosen:
        ref = GOLDEN["results"]["sequential"][r["id"]]
        out = dec.handle(r["body"])
        assert out["usage"]["input_tokens"] == ref["input_tokens"]
        for name, a in out["answers"].items():
            na, nb = numbers(a), numbers(ref["answers"][name])
            worst = max(worst, max(abs(na[k] - nb[k]) for k in nb if k != "score"))
            flips += a.get("choice") != ref["answers"][name].get("choice")
    print(f"\ntorch {b.precision} on {b.device} vs MLX 8-bit reference: {len(chosen)} requests, "
          f"max |dp| = {worst:.4f}, argmax flips = {flips}")
    assert worst < 0.25
