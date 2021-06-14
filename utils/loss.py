import torch

from torch import nn


class SSDLoss(nn.Module):
    def __init__(self, min_hard_negatives=3, lambd=1.0):
        super(SSDLoss, self).__init__()

        self.cl_loss = nn.CrossEntropyLoss(reduction='none')
        self.bb_loss = nn.SmoothL1Loss(reduction='none')
        self._min_hard_negatives = max(0, min_hard_negatives)
        self._lambd = lambd
        self._rho = 1.0

    def forward(self, pred_cls, pred_boxes, target_cls, target_boxes):
        mask = target_cls > 0
        num_pos = mask.sum(dim=1)
        # num_pos_list = [torch.empty_like(num_pos, device='cuda')] * self.num_gpus

        if num_pos.sum() < 1 and self._min_hard_negatives < 1:
            return 0, 0, 0

        box_loss = self.bb_loss(pred_boxes, target_boxes).sum(dim=1)
        box_loss = (mask.float() * box_loss).sum(dim=1)

        confidence = self.cl_loss(pred_cls, target_cls)

        # hard mining
        confidence_neg = confidence.clone()
        confidence_neg[mask] = 0
        _, neg_idx = confidence_neg.sort(dim=1, descending=True)
        _, neg_rank = neg_idx.sort(dim=1)
        neg_num = torch.min(self._min_hard_negatives * num_pos, mask.size(1) - num_pos).unsqueeze(-1)
        neg_mask = neg_rank < neg_num

        cls_loss = (confidence * (mask.float() + neg_mask.float())).sum(dim=1)

        total_loss = cls_loss + self._lambd * box_loss
        num_mask = (num_pos > 0).float()
        pos_num = num_pos.float().clamp(min=1e-6).sum()
        cls_loss = (cls_loss * num_mask / pos_num).sum()
        box_loss = (box_loss * num_mask / pos_num).sum()
        total_loss = (total_loss * num_mask / pos_num).sum()
        return cls_loss, box_loss, total_loss
