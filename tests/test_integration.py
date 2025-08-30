import pytest
from pytorch_lightning import Trainer

from data.load_data import prepare_dataloader
from model.resnet import resnet50
from model.resnet_pl import Resnet50Module


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestIntegration:
    def test_dataloader_and_resnet_model_integration(self):
        loader = prepare_dataloader(is_train=False)
        model = resnet50()

        batch = next(iter(loader))
        images, _ = batch
        logits = model(images)

        assert logits.shape[0] == images.shape[0]
        assert logits.shape == (images.shape[0], 10)

    def test_dataloader_and_resnet_pl_model_integration(self, get_real_cfg):
        loader = prepare_dataloader(is_train=False)
        cfg = get_real_cfg
        eval_cfg = cfg["eval"]
        model = Resnet50Module(cfg=eval_cfg, num_classes=10)

        batch = next(iter(loader))
        loss, acc = model(batch)
        print(type(loss), type(acc))

        assert isinstance(loss.item(), float)
        assert isinstance(acc.item(), float)

        # assert isinstance(loss, float)
        # assert isinstance(acc, float)

    def test_resnet_pl_end_to_end(self, get_real_cfg, get_fake_dataloader, tmp_path):
        cfg = get_real_cfg
        train_cfg = cfg["train"]
        eval_cfg = cfg["eval"]
        k = eval_cfg["k"]
        print(eval_cfg, k)

        model = Resnet50Module(cfg=train_cfg, num_classes=10)
        model._train_loader = get_fake_dataloader
        model._val_loader = get_fake_dataloader

        trainer = Trainer(
            max_epochs=1,
            fast_dev_run=True,
            default_root_dir=tmp_path,
            logger=False,
            enable_checkpointing=False,
        )

        # 1 step train + 1 step validate
        trainer.fit(model)

        # 1 step validate
        trainer.validate(model)

        # 1 step test
        trainer.test(model)

        assert "benchmark_avg_time_per_sample" in trainer.logged_metrics
        assert f"benchmark_top{k}_acc" in trainer.logged_metrics.keys()
