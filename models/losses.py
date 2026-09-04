"""
Loss functions for the hybrid detector.

For a capstone timeline, a full TaskAlignedAssigner (like Ultralytics
uses) is a lot of extra code for marginal benefit — this file uses a
simplified but standard center-prior + IoU assignment strategy, which
is enough to train and validate the architecture. Swap in Ultralytics'
TAL assigner later if you want to squeeze out extra mAP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def box_iou(boxes1, boxes2):
    """boxes: (N,4) and (M,4) xyxy -> (N,M) IoU"""
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


def ciou_loss(pred, target):
    """Complete IoU loss between (N,4) xyxy boxes."""
    pred = pred.float()
    target = target.float()
    iou = box_iou(pred, target).diag()
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    center_dist = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    ex1, ey1 = torch.min(px1, tx1), torch.min(py1, ty1)
    ex2, ey2 = torch.max(px2, tx2), torch.max(py2, ty2)
    diag = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

    pw, ph = (px2 - px1).clamp(min=1e-4), (py2 - py1).clamp(min=1e-4)
    tw, th = (tx2 - tx1).clamp(min=1e-4), (ty2 - ty1).clamp(min=1e-4)
    v = (4 / (torch.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - center_dist / diag - alpha * v
    return 1 - ciou


def assign_targets_simple(pred_boxes, anchors, gt_boxes, gt_classes, num_classes,
                           center_radius=2.5, stride_tensor=None):
    """
    Simplified center-prior assignment (YOLOX-style "simOTA lite"):
    an anchor is a positive candidate for a GT box if the anchor
    center falls within `center_radius` grid cells of the GT box
    center; among candidates, pick the one with highest IoU with
    that GT as its unique match (kept simple / vectorized-per-image).
    Returns per-anchor target class (-1 = background) and matched GT box.
    """
    N = anchors.shape[0]
    device = anchors.device
    target_cls = torch.full((N,), -1, dtype=torch.long, device=device)
    target_box = torch.zeros((N, 4), dtype=torch.float32, device=device)

    if gt_boxes.numel() == 0:
        return target_cls, target_box, torch.zeros(N, dtype=torch.bool, device=device)

    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

    for g in range(gt_boxes.shape[0]):
        radius_px = center_radius * stride_tensor.squeeze(-1)
        dist = ((anchors[:, 0] - gt_cx[g]) ** 2 + (anchors[:, 1] - gt_cy[g]) ** 2).sqrt()
        candidates = dist < radius_px
        if candidates.sum() == 0:
            continue
        ious = box_iou(pred_boxes[candidates], gt_boxes[g:g + 1]).squeeze(-1)
        cand_idx = candidates.nonzero(as_tuple=True)[0]
        best_local = ious.argmax()
        best_idx = cand_idx[best_local]
        target_cls[best_idx] = gt_classes[g]
        target_box[best_idx] = gt_boxes[g]

        # also mark the top-k=3 closest-by-IoU candidates positive, standard trick to give enough positive signal
        k = min(3, candidates.sum().item())
        topk_idx = cand_idx[ious.topk(k).indices]
        target_cls[topk_idx] = gt_classes[g]
        target_box[topk_idx] = gt_boxes[g]

    pos_mask = target_cls >= 0
    return target_cls, target_box, pos_mask


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, box_weight=7.5, cls_weight=1.0, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _focal_loss(self, logits, targets):
        logits = logits.float()
        targets = targets.float()
        bce = self.bce(logits, targets)
        p = logits.sigmoid()
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        return alpha_t * modulating * bce

    def forward(self, preds, gt_boxes_list, gt_classes_list):
        cls_logits = preds["cls_logits"]   # (B,N,num_classes)
        pred_boxes = preds["boxes"]        # (B,N,4)
        anchors = preds["anchors"][0]      # (N,2) pixel-space anchor centers
        strides = preds["strides"]         # (1,N,1)

        B = cls_logits.shape[0]
        total_cls_loss, total_box_loss = 0.0, 0.0

        for b in range(B):
            gt_boxes = gt_boxes_list[b].to(cls_logits.device)
            gt_classes = gt_classes_list[b].to(cls_logits.device)

            target_cls, target_box, pos_mask = assign_targets_simple(
                pred_boxes[b], anchors, gt_boxes, gt_classes,
                self.num_classes, stride_tensor=strides[0]
            )
            n_pos = max(pos_mask.sum().item(), 1)

            cls_target_onehot = torch.zeros_like(cls_logits[b])
            if pos_mask.any():
                cls_target_onehot[pos_mask, target_cls[pos_mask]] = 1.0

            cls_loss = self._focal_loss(cls_logits[b], cls_target_onehot).sum() / n_pos

            if pos_mask.any():
                box_loss = ciou_loss(pred_boxes[b][pos_mask], target_box[pos_mask]).sum() / n_pos
            else:
                box_loss = torch.tensor(0.0, device=cls_logits.device)

            total_cls_loss += cls_loss
            total_box_loss += box_loss

        total_cls_loss /= B
        total_box_loss /= B
        loss = self.cls_weight * total_cls_loss + self.box_weight * total_box_loss
        return {
            "loss": loss,
            "cls_loss": total_cls_loss.detach(),
            "box_loss": total_box_loss.detach() if torch.is_tensor(total_box_loss) else total_box_loss,
        }


class AuxLosses(nn.Module):
    """Drivable-area segmentation CE + weather/timeofday CE."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, drivable_target, weather_target, timeofday_target):
        losses = {}
        if "drivable_logits" in preds:
            losses["drivable_loss"] = self.ce(preds["drivable_logits"], drivable_target)
        if "weather_logits" in preds:
            losses["weather_loss"] = self.ce(preds["weather_logits"], weather_target)
        if "timeofday_logits" in preds:
            losses["timeofday_loss"] = self.ce(preds["timeofday_logits"], timeofday_target)
        return losses
