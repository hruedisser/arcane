import torch
from torchmetrics import Metric


def weighted_mse_loss(input, target, weight):
    return torch.sum(
        weight * ((input - target) ** 2)[:, 0] + weight * ((input - target) ** 2)[:, 1]
    )


def calculate_confusion_matrix(true_labels, pred_labels):
    # Convert float tensors to binary (0 or 1)
    true_labels = (
        true_labels > 0.5
    ).float()  # Assuming threshold of 0.5 for binary classification
    pred_labels = (
        pred_labels > 0.5
    ).float()  # Assuming threshold of 0.5 for binary classification

    # Initialize counts
    TP = ((true_labels == 1) & (pred_labels == 1)).sum().item()  # True Positive
    FP = ((true_labels == 0) & (pred_labels == 1)).sum().item()  # False Positive
    TN = ((true_labels == 0) & (pred_labels == 0)).sum().item()  # True Negative
    FN = ((true_labels == 1) & (pred_labels == 0)).sum().item()  # False Negative

    return TP, FP, TN, FN


# Heidke Skill Score (HSS)
class HSS(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("TP", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("FP", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("TN", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("FN", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, true_labels, pred_labels):
        TP, FP, TN, FN = calculate_confusion_matrix(true_labels, pred_labels)
        self.TP += TP
        self.FP += FP
        self.TN += TN
        self.FN += FN

    def compute(self):
        if (self.TP == 0) & (self.FP == 0) & (self.FN == 0):
            return torch.tensor(1)

        obs_sum_poss = self.TP + self.FP
        obs_sum_negs = self.FN + self.TN

        fcst_sum_poss = self.TP + self.FN
        fcst_sum_negs = self.FP + self.TN

        HSS = (
            2
            * (self.TP * self.TN - self.FP * self.FN)
            / (fcst_sum_poss * obs_sum_negs + obs_sum_poss * fcst_sum_negs)
        )
        return HSS


class TSS(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("TP", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("FP", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("TN", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("FN", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, true_labels, pred_labels):

        TP, FP, TN, FN = calculate_confusion_matrix(true_labels, pred_labels)
        self.TP += TP
        self.FP += FP
        self.TN += TN
        self.FN += FN

    def compute(self):

        if (self.TP == 0) & (self.FP == 0) & (self.FN == 0):
            return torch.tensor(1)

        TPR = self.TP / (self.TP + self.FN)
        FPR = self.FP / (self.FP + self.TN)
        TSS = TPR - FPR

        return TSS
