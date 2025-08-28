import torch
from torch import nn
from torchvision.models import resnet50
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from model.scheduler import WarmupCosineLR

class Resnet50Module(pl.LightningModule):
    def __init__(self, num_classes, cfg, pretrained_path=None):
        super().__init__()
        assert cfg is not None, "Config cannot be None"
        self.cfg = cfg
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.model = resnet50(weights=None)        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        if pretrained_path:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_state_dict = torch.load(pretrained_path, map_location=device)
            self.model.load_state_dict(model_state_dict)
    
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

    def test_step(self, batch):
        _, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

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

def load_resnet():
    import os
    model = resnet50(weights=None)
    pretrained_path = os.path.join("E:/ResNet-Compress/retrain/pretrained_wt_resnet50", "resnet50.pt")
    print(list(torch.load(pretrained_path).keys())[:10])

    # traced = symbolic_trace(model)

    # for node in traced.graph.nodes:
    #     print(f"{node.op}: {node.name} -> {node.target}")

    # traced.fc = nn.Linear()

    return model

if __name__ == "__main__":
    resnet_model = load_resnet()