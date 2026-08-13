from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EvalResult:
    balanced_acc: float
    tpr: float
    tnr: float
    auc: float
    threshold: float
    best_threshold: float
    best_balanced_acc: float
    per_source: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"balanced_acc@{self.threshold:.2f}: {self.balanced_acc:.4f} "
            f"(TPR {self.tpr:.4f} / TNR {self.tnr:.4f}), AUC {self.auc:.4f}",
            f"best threshold {self.best_threshold:.3f} -> {self.best_balanced_acc:.4f}",
        ]
        lines += [f"  {src}: {acc:.4f}" for src, acc in sorted(self.per_source.items())]
        return "\n".join(lines)


class Evaluator:
    def __init__(self, threshold: float = 0.65, device: str = "cuda"):
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def collect(self, model, dataset, batch_size=64, workers=8) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
        model.eval().to(self.device)
        probs, labels = [], []
        for x, y in tqdm(loader, desc="eval", leave=False):
            p = model.predict_proba(x.to(self.device, non_blocking=True))
            probs.append(p.float().cpu().numpy())
            labels.append(y.numpy())
        return np.concatenate(probs), np.concatenate(labels)

    def score(self, probs: np.ndarray, labels: np.ndarray,
              sources: list[str] | None = None) -> EvalResult:
        from sklearn.metrics import roc_auc_score

        fake, real = labels == 1, labels == 0
        tpr = float((probs[fake] >= self.threshold).mean()) if fake.any() else 0.0
        tnr = float((probs[real] < self.threshold).mean()) if real.any() else 0.0
        auc = float(roc_auc_score(labels, probs)) if fake.any() and real.any() else 0.0

        ths = np.linspace(0.01, 0.99, 197)
        baccs = [((probs[fake] >= t).mean() + (probs[real] < t).mean()) / 2 for t in ths]
        best = int(np.argmax(baccs))

        per_source = {}
        if sources is not None:
            src = np.array(sources)
            for s in np.unique(src):
                m = src == s
                correct = (probs[m] >= self.threshold) == (labels[m] == 1)
                per_source[str(s)] = float(correct.mean())

        return EvalResult(
            balanced_acc=(tpr + tnr) / 2, tpr=tpr, tnr=tnr, auc=auc,
            threshold=self.threshold,
            best_threshold=float(ths[best]), best_balanced_acc=float(baccs[best]),
            per_source=per_source,
        )

    def run(self, model, dataset, batch_size=64) -> EvalResult:
        probs, labels = self.collect(model, dataset, batch_size)
        sources = [s.source for s in dataset.samples]
        return self.score(probs, labels, sources)
