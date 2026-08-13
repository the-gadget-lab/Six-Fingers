from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .eval import Evaluator


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 1e-4
    head_lr_mult: float = 10.0
    weight_decay: float = 0.05
    batch_size: int = 64
    workers: int = 12
    warmup_frac: float = 0.05
    patience: int = 2
    out_dir: Path = Path("checkpoints")


class Trainer:
    def __init__(self, model, head: torch.nn.Module, train_ds, val_ds, cfg: TrainConfig, device="cuda"):
        self.model = model.to(device)
        self.head = head
        self.cfg = cfg
        self.device = device
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.evaluator = Evaluator(threshold=0.65, device=device)

    def _optimizer(self):
        head_params, body_params = [], []
        head_ids = {id(p) for p in self.head.parameters()}
        for p in self.model.parameters():
            (head_params if id(p) in head_ids else body_params).append(p)
        return torch.optim.AdamW(
            [{"params": body_params, "lr": self.cfg.lr},
             {"params": head_params, "lr": self.cfg.lr * self.cfg.head_lr_mult}],
            weight_decay=self.cfg.weight_decay,
        )

    def run(self) -> float:
        cfg = self.cfg
        loader = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.workers, pin_memory=True, drop_last=True,
                            persistent_workers=True)
        opt = self._optimizer()
        total_steps = len(loader) * cfg.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=total_steps,
            pct_start=cfg.warmup_frac)
        scaler = torch.amp.GradScaler()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        cfg.out_dir.mkdir(parents=True, exist_ok=True)

        best_bacc = 0.0
        stale = 0
        for epoch in range(cfg.epochs):
            self.model.train()
            bar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
            for x, y in bar:
                x = x.to(self.device, non_blocking=True)
                y = y.float().to(self.device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_fn(self.model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                bar.set_postfix(loss=f"{loss.item():.4f}")

            res = self.evaluator.run(self.model, self.val_ds, cfg.batch_size)
            print(f"[epoch {epoch + 1}]\n{res.summary()}")
            if res.balanced_acc > best_bacc:
                best_bacc = res.balanced_acc
                stale = 0
                torch.save(self.model.state_dict(), cfg.out_dir / "best.pt")
                print(f"saved best.pt ({best_bacc:.4f})")
            else:
                stale += 1
                if stale >= cfg.patience:
                    print(f"early stop: no val improvement for {stale} epochs")
                    break
        return best_bacc
