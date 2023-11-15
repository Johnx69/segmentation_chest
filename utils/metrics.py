import torch
import pandas as pd
import os


def calculate_overlap_metrics(gt, pred, eps=1e-5):
    output = pred.view(
        -1,
    )
    target = gt.view(
        -1,
    ).float()

    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN

    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    #     specificity = (tn + eps) / (tn + fp + eps)

    return pixel_acc, dice, iou, precision, recall


def calculate_overlap_metrics_post(gt, pred, eps=1e-5):
    output = (
        pred.view(
            -1,
        )
        / 255.0
    )
    target = (
        gt.view(
            -1,
        )
        / 255.0
    )

    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN

    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    #     specificity = (tn + eps) / (tn + fp + eps)

    return pixel_acc, dice, iou, precision, recall


def calculate_f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def export_to_excel(filename, results):
    # Define the column names

    for i in range(3, len(results) - 1):
        results[i] = results[i].cpu().numpy()
        results[i] = round(results[i] * 100, 2)

    if os.path.exists(filename):
        df = pd.read_excel(filename)
    else:
        columns = [
            "Condition",
            "Backbone",
            "Classification F1 Score",
            "Lung segmentation F1 Score",
            "Lung segmentation IoU",
            "Lung segmentation Dice",
            "Infection segmentation F1 Score",
            "Infection segmentation IoU",
            "Infection segmentation Dice",
            "Mean F1",
            "#param"
        ]

        # Create an empty DataFrame with the specified columns
        df = pd.DataFrame(columns=columns)

    # Append the new row to the existing DataFrame
    df.loc[len(df)] = results
    # Write the updated DataFrame to an Excel file
    df.to_excel(filename, index=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
