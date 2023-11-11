import torch


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
