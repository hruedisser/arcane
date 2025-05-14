import numpy as np
import torch

## Utils for IoU Losses


def get_boxes_tensor(output_tensor, device="cuda"):
    """
    input: tensor (Num_classes, num_cells, (t, w, p))
    output: tensor(num_classes, num_cells, (coord_begin, coord_end, p))
    made with the consideration of icmes are detecting according to beginning date and CIR according to center date
    """
    boxes_tensor = torch.tensor(np.zeros(output_tensor.shape)).to(device)

    num_cells = output_tensor.shape[2]

    boxes_tensor[..., 2] = output_tensor[..., 2]
    boxes_tensor[:, 0, :, 0] = output_tensor[:, 0, :, 0] + torch.tensor(
        range(num_cells)
    ).to(device)
    boxes_tensor[:, 1, :, 0] = (
        output_tensor[:, 1, :, 0]
        + torch.tensor(range(num_cells)).to(device)
        - 0.5 * output_tensor[:, 1, :, 1] * torch.tensor(num_cells).to(device)
    )

    boxes_tensor[:, 0, :, 1] = (
        output_tensor[:, 0, :, 0]
        + torch.tensor(range(num_cells)).to(device)
        + output_tensor[:, 0, :, 1] * torch.tensor(num_cells).to(device)
    )
    boxes_tensor[:, 1, :, 1] = (
        output_tensor[:, 1, :, 0]
        + torch.tensor(range(num_cells)).to(device)
        + 0.5 * output_tensor[:, 1, :, 1] * torch.tensor(num_cells).to(device)
    )

    return boxes_tensor


def inter(eventA, eventB):
    return torch.max(
        torch.min(eventA[..., 1], eventB[..., 1])
        - torch.max(eventA[..., 0], eventB[..., 0]),
        torch.tensor(0.0),
    )


def union(eventA, eventB):
    return (
        eventA[..., 1]
        + eventB[..., 1]
        - eventA[..., 0]
        - eventB[..., 0]
        - inter(eventA, eventB)
        + torch.tensor(1e-10)
    )


def Convex_union(eventA, eventB):
    return (
        torch.max(eventA[..., 1], eventB[..., 1])
        - torch.min(eventA[..., 0], eventB[..., 0])
        + torch.tensor(1e-10)
    )


def IoU(eventA, eventB):
    return inter(eventA, eventB) / union(eventA, eventB)


def GIoU(eventA, eventB):
    C = Convex_union(eventA, eventB)
    return IoU(eventA, eventB) - (C - union(eventA, eventB)) / C


def center_distance(eventA, eventB):
    centerA = eventA[..., 0] + 0.5 * (eventA[..., 1] - eventA[..., 0])
    centerB = eventB[..., 0] + 0.5 * (eventB[..., 1] - eventB[..., 0])
    return centerA - centerB


def DIoU(eventA, eventB):
    C = Convex_union(eventA, eventB)
    return IoU(eventA, eventB) - (center_distance(eventA, eventB) / C) ** 2


def GIoU_Loss(output, target):
    return 1 - GIoU(output, target)


def DIoU_Loss(output, target):
    return 1 - DIoU(output, target)


def yoloss_GIoU(output, target, thres_obj=0.5, l_obj=5.0, l_box=5.0, l_no_obj=0.5):
    obj = target[..., 2]
    obj[obj < thres_obj] = 0

    obj_loss = torch.sum(obj * GIoU_Loss(output, target))
    conf_loss = torch.sum(
        torch.unsqueeze(obj, -1) * torch.square(output[..., 2:] - target[..., 2:])
    )
    no_obj_loss = torch.sum(
        torch.unsqueeze(1 - target[..., 2], -1)
        * torch.square(output[..., 2:] - target[..., 2:])
    )

    return l_box * obj_loss + l_obj * conf_loss + l_no_obj * no_obj_loss


def dice_coeff(y_pred, y_true):
    intersection = torch.sum(y_true * y_pred[:, 1, :])
    return (2 * intersection + 1) / (torch.sum(y_true) + torch.sum(y_pred[:, 1, :]) + 1)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)
