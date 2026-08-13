import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.cf import CLIP_MEAN, CLIP_STD, CFPreprocess
from slop.data import SlopDataset, WebRealistic
from slop.detector import HFDetector
from slop.eval import Evaluator

OUT = Path("../model/dist")


def calibrate(model, ds, batch: int) -> float:
    """Bias b such that sigmoid(logit+b) >= 0.65 exactly at the balanced-accuracy-optimal raw threshold."""
    ev = Evaluator(threshold=0.65)
    probs, labels = ev.collect(model, ds, batch_size=batch, workers=8)
    res = ev.score(probs, labels)
    t = res.best_threshold
    bias = float(np.log(0.65 / 0.35) - np.log(t / (1 - t)))
    print(f"pre-calibration: {res.summary()}")
    print(f"raw t*={t:.4f} -> logit bias {bias:+.4f}")
    ev2 = Evaluator(threshold=0.65)
    z = np.log(probs / np.clip(1 - probs, 1e-9, None) + 1e-12) + bias
    res2 = ev2.score(1 / (1 + np.exp(-z)), labels)
    print(f"post-calibration: {res2.summary()}")
    return bias


class ExportWrapper(torch.nn.Module):
    def __init__(self, detector: HFDetector):
        super().__init__()
        self.detector = detector

    def forward(self, pixel_values):
        return self.detector(pixel_values).unsqueeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--val-root", default="../data/cf_eval")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--skip-quant", action="store_true")
    args = ap.parse_args()

    model = (HFDetector.from_checkpoint(args.ckpt) if args.ckpt else HFDetector()).cuda().eval()
    val = SlopDataset(args.val_root, transform=CFPreprocess(), degrade=WebRealistic(seed=0))
    model.logit_bias = calibrate(model, val, args.batch)

    OUT.mkdir(parents=True, exist_ok=True)
    fp32_path = OUT / "model_fp32.onnx"
    model = model.cpu().eval()
    x = torch.randn(1, 3, 384, 384)
    torch.onnx.export(
        ExportWrapper(model), (x,), fp32_path, input_names=["pixel_values"],
        output_names=["logits"], dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    import onnxruntime as rt

    with torch.no_grad():
        ref = model(x).numpy()
    sess = rt.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixel_values": x.numpy()})[0].squeeze(-1)
    diff = float(np.abs(got - ref).max())
    print(f"fp32 onnx parity: max|diff|={diff:.2e}")
    assert diff < 1e-3

    final = fp32_path
    if not args.skip_quant:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q_path = OUT / "model.onnx"
        quantize_dynamic(fp32_path, q_path, weight_type=QuantType.QUInt8)
        final = q_path
        sess_q = rt.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
        got_q = sess_q.run(None, {"pixel_values": x.numpy()})[0].squeeze(-1)
        print(f"int8 vs torch on random input: {float(np.abs(got_q - ref).max()):.4f}")

    spec = {
        "file": "model.onnx", "inputSize": 384, "resizeSize": 440,
        "mean": list(CLIP_MEAN), "std": list(CLIP_STD), "threshold": 0.65,
    }
    (OUT / "model.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"exported {final} ({final.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
