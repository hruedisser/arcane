import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pytorch_lightning import Callback
from torchvision.utils import make_grid


class VisualisationCallbackSegmentation(Callback):
    def __init__(
        self,
        image_key: str = "insitu",
        label_key: str = "catalog",
        num_samples: int = 25,
        classification: bool = True,
    ):
        super().__init__()

        self.image_key = image_key
        self.label_key = label_key
        self.num_samples = num_samples
        self.colors = ["#c20078", "#f97306", "#069af3", "#000000"]
        self.colorscale = "Purples"
        self.classification = classification

    def on_train_epoch_end(self, trainer, pl_module):

        dataloader = trainer.train_dataloader

        image_grid, label_str = self.sample_from_dataset(
            dataloader,
            pl_module,
            self.num_samples,
        )

        trainer.logger.log_image(
            "train/images", [image_grid], trainer.global_step, caption=[label_str]
        )

    def on_validation_epoch_end(self, trainer, pl_module):

        data_loader = trainer.val_dataloaders

        image_grid, label_str = self.sample_from_dataset(
            data_loader,
            pl_module,
            self.num_samples,
        )

        trainer.logger.log_image(
            "val/images", [image_grid], trainer.global_step, caption=[label_str]
        )

    def sample_from_dataset(self, dataloader, model, n_images):

        images = []
        label_str = []

        for batch in dataloader:

            sample = batch[self.image_key]
            y = batch[self.label_key]

            with torch.no_grad():
                y_hat = model(sample.to(model.device)).cpu()

            if self.classification:
                if len(y.shape) > 1:
                    if y.shape[1] > 1:
                        if model.mode == "last":
                            y_single = y[:, -1]
                        elif model.mode == "max":
                            y_single = torch.argmax(y, dim=1)

                y_hat_bin = torch.argmax(y_hat, dim=1)[:, -1]
                y_num = y_hat[:, 1, -1]

            for i in range(min(n_images, sample.size(0))):
                fig, axes = plt.subplots(3, 1, figsize=(6, 9))

                for var in range(4):
                    axes[0].plot(
                        sample[i, :, var], color=self.colors[var], label=f"Var {var+1}"
                    )

                # Plot true label as a heatmap in the second subplot

                if len(y[i].shape) == 1:
                    sns.heatmap(
                        y[i].unsqueeze(0).numpy(),
                        cmap=self.colorscale,
                        ax=axes[1],
                        cbar=False,
                    )
                else:
                    sns.heatmap(
                        y[i].numpy(), cmap=self.colorscale, ax=axes[1], cbar=False
                    )

                if len(y_hat[i].shape) == 1:
                    sns.heatmap(
                        y_hat[i].unsqueeze(0).numpy(),
                        cmap=self.colorscale,
                        ax=axes[2],
                        cbar=False,
                    )
                else:
                    sns.heatmap(
                        y_hat[i][1:].numpy(),
                        cmap=self.colorscale,
                        ax=axes[2],
                        cbar=False,
                    )

                if self.classification:  # Set the title with true and predicted label

                    true_label = y_single[i].item()
                    predicted_label = y_num[i].item()
                    binary_label = y_hat_bin[i].item()

                    title = f"True: {true_label}, Predicted: {predicted_label:.2f}"
                    axes[0].set_title(title)

                    # Set border color based on the match
                    if true_label == binary_label:
                        color = "green"
                    else:
                        color = "red"

                    # Draw border
                    for ax in axes:
                        for spine in ax.spines.values():
                            spine.set_edgecolor(color)
                            spine.set_linewidth(6)
                else:
                    title = f"Image {i}"

                # Store the image
                fig.canvas.draw()
                np_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                np_img = np_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(torch.tensor(np_img).permute(2, 0, 1))

                label_str.append("")
                plt.close(fig)

            if len(images) >= n_images:
                break

        # Determine grid size (e.g., 5x5 for 25 images)
        grid_size = int(math.sqrt(n_images))

        # Create a grid of images
        image_grid = make_grid(images[:n_images], nrow=grid_size)
        return image_grid, label_str[:n_images]
