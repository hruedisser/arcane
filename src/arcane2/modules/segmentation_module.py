import torch
import torch.nn as nn
import torchmetrics
from loguru import logger
from pytorch_lightning import LightningModule

from .lambda_scheduler import lambda_scheduler
from .loss_utils import dice_loss
from .metrics import HSS, TSS

"""
Module for training a segmentation model.
"""


class SegmentationModule(LightningModule):
    def __init__(
        self,
        model,
        num_classes: int = 2,
        lr: float = 1e-3,
        class_weights=None,
        scheduler: dict = None,
        optimizer: dict = {},
        classification: bool = True,
        mode="last",
        loss="cross_entropy",
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

        if loss == "cross_entropy":
            self.loss_ce = nn.CrossEntropyLoss(weight=class_weights)
        elif loss == "dice":
            self.loss_ce = dice_loss

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.lr = lr
        self.scheduler = scheduler
        self.optimizer = optimizer

        if classification:
            self.hss = nn.ModuleDict({"_train": HSS(), "_val": HSS(), "_test": HSS()})
            self.tss = nn.ModuleDict({"_train": TSS(), "_val": TSS(), "_test": TSS()})

        self.classification = classification

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
                batch_size=batch_size,
            )
        if where == "val":
            self.log(
                "val_loss",
                loss["loss"].detach(),
                on_epoch=True,
                on_step=False,
                logger=True,
                batch_size=batch_size,
            )

    def common_step(self, batch, batch_idx, step_name):
        """
        Common training/validation/testing step for segmentation.

        Batch is a dictionary containing the following keys:
        ["idx", "timestamp", ...]

        step_name: str
            Name of the step (train, val, test)
        """

        insitu_data = batch["insitu"].float()
        y = batch["catalog"].float()

        y_hat = self.model(insitu_data).float()

        loss_cross_entropy = self.loss_ce(y_hat, y.long())
        accuracy = self.accuracy(y_hat, y.long())

        loss = {f"loss_ce": loss_cross_entropy, "accuracy": accuracy}
        loss[f"loss"] = loss[f"loss_ce"]

        self.log_losses(loss, step_name, insitu_data.shape[0])

        if self.classification:
            if self.mode == "last":
                y_classification = y[:, -1]
                y_hat_classification = y_hat.argmax(dim=-2).float()[:, -1]
            elif self.mode == "max":
                y_classification = torch.argmax(y, dim=1)
                y_hat_classification = torch.argmax(y_hat, dim=1)

            self.hss[f"_{step_name}"].update(y_hat_classification, y_classification)
            self.tss[f"_{step_name}"].update(y_hat_classification, y_classification)

            self.log(
                f"{step_name}/HSS",
                self.hss[f"_{step_name}"].compute(),
                on_epoch=True,
                logger=True,
            )

            self.log(
                f"{step_name}/TSS",
                self.tss[f"_{step_name}"].compute(),
                on_epoch=True,
                logger=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx, "train")
        return loss_dict

    def validation_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx, "val")
        return loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx, "test")
        return loss_dict

    def configure_optimizers(self):
        # Choose the optimizer based on the configuration
        optimizer_type = self.optimizer.get(
            "type", "Adam"
        )  # Default to Adam if not specified
        optimizer_params = self.optimizer.get(
            "params", {}
        )  # Additional parameters for the optimizer

        # Dynamically get the optimizer class and initialize it
        optimizer_class = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_class(
            self.model.parameters(), lr=self.lr, **optimizer_params
        )

        logger.info(f"Using optimizer: {optimizer}")

        # Check if scheduler is provided in the configuration
        if self.scheduler:

            scheduler_type = self.scheduler.get("type")
            scheduler_params = self.scheduler.get("params", {})

            if scheduler_type == "LambdaLR":
                ## Map the string to the actual function
                lr_lambda_name = scheduler_params.get("lr_lambda")
                lr_lambda_callable = None

                if lr_lambda_name == "lambda_scheduler":
                    lr_lambda_callable = lambda_scheduler  # Use the imported function

                if lr_lambda_callable is None or not callable(lr_lambda_callable):
                    raise ValueError(
                        f"For LambdaLR, 'lr_lambda' must be a callable. Got: {type(lr_lambda_callable)}"
                    )

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lr_lambda_callable
                )
            else:
                # Dynamically initialize other schedulers
                scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(
                    optimizer, **scheduler_params
                )

            logger.info(f"Using scheduler: {scheduler}")

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
