import torch

from model.resnet import resnet50
from model.resnet_pl import Resnet50Module


class TestForward:
    def test_resnet_forward(self, create_fake_batch):
        images, _ = create_fake_batch
        model = resnet50()
        logits = model(images)
        assert logits is not None
        assert logits.shape == (2, 10)

    def test_resnet_pl_forward(self, create_fake_batch, get_real_cfg):
        images, labels = create_fake_batch
        cfg = get_real_cfg
        assert cfg is not None, "Config failed to load"
        train_cfg = cfg["train"]
        model = Resnet50Module(
            num_classes=10, cfg=train_cfg, pretrained_path="model/weights"
        )

        model.eval()
        with torch.no_grad():
            loss, acc = model((images, labels))

        loss, acc = model((images, labels))
        assert loss is not None
        assert acc is not None
