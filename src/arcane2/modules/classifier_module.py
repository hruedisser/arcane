import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from .metrics import HSS, TSS

"""
Use this module as base classifier module.
"""


class ClassifierModule(LightningModule):
    def __init__(
        self,
        model,
        num_classes: int = 2,
        lr: float = 1e-3,
        class_weights=None,
        scheduler: dict = None,
        mode="last",
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model = model

        assert num_classes > 0, "Number of classes must be greater than 0"
        self.num_classes = num_classes

        self.mode = mode

        if class_weights is not None:
            assert (
                len(class_weights) == num_classes
            ), "Number of class weights must match number of classes"
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        if num_classes == 1:
            self.loss_ce = nn.BCEWithLogitsLoss(weight=class_weights)
        else:
            self.loss_ce = nn.CrossEntropyLoss(weight=class_weights)

        self.lr = lr
        self.scheduler = scheduler

        self.hss = nn.ModuleDict({"_train": HSS(), "_val": HSS(), "_test": HSS()})
        self.tss = nn.ModuleDict({"_train": TSS(), "_val": TSS(), "_test": TSS()})

    def forward(self, x):
        return self.model(x)

    def log_losses(self, loss, where, batch_size):
        for k in loss.keys():
            self.log(
                f"{where}/{k}",
                loss[k].detach(),
                on_epoch=True,
                on_step=False,
                logger=True,
                # sync_dist=True,
                batch_size=batch_size,
            )

    def common_step(self, batch, batch_idx, step_name):
        """
        Common training/validation step for the flare prediction module.

        Batch is a dictionary containing the following keys:
        ['idx', 'timestamp', ...]

        step_name:str is a name of the training step, i.e. train, val, test
        """

        insitu_data = batch["insitu"]
        y = batch["catalog"]

        y_hat = self.model(insitu_data)

        if len(y.shape) > 1:
            if y.shape[1] > 1:
                if self.mode == "last":
                    y = y[:, -1]

                elif self.mode == "max":
                    y = torch.argmax(y, dim=1)

        if self.num_classes > 1:
            y_oh = F.one_hot(y.long(), num_classes=self.num_classes).float()

            loss_cross_entropy = self.loss_ce(y_hat, y_oh)
        else:
            loss_cross_entropy = self.loss_ce(y_hat, y)

        loss = {f"loss_ce": loss_cross_entropy}
        loss[f"loss"] = loss[f"loss_ce"]
        self.log_losses(loss, step_name, insitu_data.shape[0])

        if self.num_classes > 1:
            y_hat = torch.argmax(y_hat, dim=1)

        self.hss[f"_{step_name}"].update(y_hat, y)
        self.tss[f"_{step_name}"].update(y_hat, y)

        self.log(
            f"{step_name}/HSS",
            self.hss[f"_{step_name}"].compute(),
            on_epoch=True,
            logger=True,
            # sync_dist=True,
        )

        self.log(
            f"{step_name}/TSS",
            self.tss[f"_{step_name}"].compute(),
            on_epoch=True,
            logger=True,
            # sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx, "train")
        return loss_dict

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Check if a scheduler is provided in the configuration
        if self.scheduler:

            scheduler_type = self.scheduler.get("type")
            scheduler_params = self.scheduler.get("params", {})

            scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(
                optimizer, **scheduler_params
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler.get("monitor", "val/loss"),
                    "frequency": 1,
                },
            }
        else:
            return optimizer
