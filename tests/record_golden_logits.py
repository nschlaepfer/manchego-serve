"""Record the option-code logits behind golden_mlx8bit.json (not a test; run once, on Apple silicon).

    python tests/record_golden_logits.py /path/to/Manchego-MLX-8bit     # oraculumai/Manchego-MLX-8bit @ 79e55e2d

For each of the 20 golden requests and both executions, this scores the request with the MLX backend and keeps each
question's option-code logits. It then checks that the v0.1.0 readout (tests/v010_reference.py) turns those logits into
the golden answers byte for byte (json.dumps equal), and writes fixtures/golden_logits_mlx8bit.json. With that file,
test_temperature.py can check the T = 1 readout against the v0.1.0 golden fixtures byte for byte without the weights.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from conftest import BY_ID, load  # noqa: E402
import v010_reference as V010  # noqa: E402
from manchego_serve.backends.mlx_backend import MLXBackend  # noqa: E402
from manchego_serve.decider import Decider  # noqa: E402
from manchego_serve.weights import PINS, sha256_file  # noqa: E402


class Recording(Decider):
    def score_many(self, encoded):
        out, calls = super().score_many(encoded)
        self.last = out
        return out, calls


def main(model_dir: str) -> None:
    pin = PINS["oraculumai/Manchego-MLX-8bit"]
    sha = sha256_file(Path(model_dir) / "model.safetensors")
    assert sha == pin["files"]["model.safetensors"], f"not the pinned weights: {sha}"
    golden = load("golden_mlx8bit.json")
    b = MLXBackend(model_dir)
    res: dict = {}
    for execution in ("sequential", "batched"):
        d = Recording(b, execution=execution)
        res[execution] = {}
        for rid, ref in golden["results"][execution].items():
            body = BY_ID[rid]["body"]
            d.handle(body)
            names = list(body["questions"])
            logits = dict(zip(names, d.last))
            for n in names:
                q = body["questions"][n]
                opts = d.plan(body["state"], q)["opts"]
                got = V010.answer(q, opts, logits[n])
                assert json.dumps(got) == json.dumps(ref["answers"][n]), (execution, rid, n)
            res[execution][rid] = {"logits": logits}
    out = {"note": "Option-code logits of oraculumai/Manchego-MLX-8bit@79e55e2d (this package's MLX backend) on the 20 golden "
                   "requests. The v0.1.0 readout turns them into golden_mlx8bit.json's answers byte for byte.",
           "weights": golden["weights"], "runtime": b.runtime, "results": res}
    (HERE / "fixtures" / "golden_logits_mlx8bit.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", HERE / "fixtures" / "golden_logits_mlx8bit.json")


if __name__ == "__main__":
    main(sys.argv[1])
