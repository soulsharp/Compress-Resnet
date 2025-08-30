import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from model.resnet import resnet50
from model.scheduler import WarmupCosineLR
from data.load_data import prepare_dataloader
from benchmarks.benchmark import run_benchmark


class Resnet50Module(pl.LightningModule):
    def __init__(self, cfg, num_classes, pretrained_path="model/weights"):
        super().__init__()
        assert cfg is not None, "Config cannot be None"
        self.cfg = cfg
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = resnet50(
            pretrained=True, pretrained_dir=pretrained_path, device=device
        )
        if num_classes != 10:
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if not hasattr(self, "_train_loader"):
            self._train_loader = prepare_dataloader(True)
        return self._train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if not hasattr(self, "_val_loader"):
            self._val_loader = prepare_dataloader(True)
        return self._val_loader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.val_dataloader()

    def test_step(self, batch):
        return None

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
            momentum=self.cfg["momentum"],
            nesterov=self.cfg["nesterov"],
        )
        total_steps = self.cfg["max_epochs"] * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    def on_test_start(self):
        k = self.cfg["k"]
        acc, avg_time = run_benchmark(self.model, self._val_loader, k)
        self.log(f"benchmark_top{k}_acc", acc, prog_bar=True)
        self.log("benchmark_avg_time_per_sample", avg_time)
        print(
            f"Top-{k} Benchmark Accuracy: {acc:.4f}, Avg time/sample: {avg_time:.6f}s"
        )
