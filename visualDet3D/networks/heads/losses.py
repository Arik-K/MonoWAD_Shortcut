
from easydict import EasyDict
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from visualDet3D.networks.utils.utils import calc_iou
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.utils.timer import profile

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, balance_weights=torch.tensor([1.0], dtype=torch.float)):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.register_buffer("balance_weights", balance_weights)

    def forward(self, classification:torch.Tensor, 
                      targets:torch.Tensor, 
                      gamma:Optional[float]=None, 
                      balance_weights:Optional[torch.Tensor]=None)->torch.Tensor:
        if gamma is None:
            gamma = self.gamma
        if balance_weights is None:
            balance_weights = self.balance_weights

        probs = torch.sigmoid(classification) #[B, N, 1]
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - probs, probs)
        focal_weight = torch.pow(focal_weight, gamma)

        bce = -(targets * nn.functional.logsigmoid(classification)) * balance_weights - ((1-targets) * nn.functional.logsigmoid(-classification)) #[B, N, 1]
        cls_loss = focal_weight * bce

        ## neglect  0.3 < iou < 0.4 anchors
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

        ## clamp over confidence and correct ones to prevent overfitting
        cls_loss = torch.where(torch.lt(cls_loss, 1e-5), torch.zeros(cls_loss.shape).cuda(), cls_loss) 

        return cls_loss

class ModifiedSmoothL1Loss(nn.Module):
    def __init__(self, L1_regression_alpha:float):
        super(ModifiedSmoothL1Loss, self).__init__()
        self.alpha = L1_regression_alpha

    def forward(self, normed_targets:torch.Tensor, pos_reg:torch.Tensor):
        regression_diff = torch.abs(normed_targets - pos_reg) #[K, 12]
        ## Smoothed-L1 formula:
        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / self.alpha),
            0.5 * self.alpha * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / self.alpha
        )
        ## clipped to avoid overfitting
        regression_loss = torch.where(
            torch.le(regression_diff, 0.01),
            torch.zeros_like(regression_loss),
            regression_loss
        )
        return regression_loss

class Loss(nn.Module):
    def __init__(self, conf):
        super(Loss, self).__init__()
        self.conf = conf
        self.anchors = Anchors(conf.anchors_cfg)
        self.cls_loss = SigmoidFocalLoss(gamma=conf.focal_loss_gamma, balance_weights=torch.tensor(conf.balance_weight).float())
        self.reg_loss = ModifiedSmoothL1Loss(conf.L1_regression_alpha)
        
        # Output Layers to convert features (256) -> Predictions (Classes + Box Reg)
        layer_cfg = conf.layer_cfg
        self.cls_layer = nn.Conv2d(layer_cfg.num_features_in, layer_cfg.num_cls_output * len(conf.anchors_cfg.ratios) * len(conf.anchors_cfg.scales), kernel_size=1)
        self.reg_layer = nn.Conv2d(layer_cfg.num_features_in, layer_cfg.num_reg_output * len(conf.anchors_cfg.ratios) * len(conf.anchors_cfg.scales), kernel_size=1)

    def forward(self, features, P2, annotations):
        # 1. Generate predictions from the feature map
        classification = self.cls_layer(features)
        regression = self.reg_layer(features)

        # 2. Reshape predictions
        batch_size = classification.shape[0]
        classification = classification.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.conf.layer_cfg.num_cls_output)
        regression = regression.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.conf.layer_cfg.num_reg_output)

        # 3. Generate Anchors
        anchors = self.anchors(features, P2) # [B, N, 4]

        # 4. Match Anchors to Ground Truth (Assign Targets)
        cls_targets, reg_targets, pos_mask, neg_mask = self.build_targets(anchors, annotations)

        # 5. Calculate Losses
        # Classification Loss
        cls_loss = self.cls_loss(classification, cls_targets)
        classification_loss = cls_loss.sum() / (pos_mask.sum() + 1e-6)

        # Regression Loss (Only for positive anchors)
        if pos_mask.sum() > 0:
            reg_loss = self.reg_loss(reg_targets[pos_mask], regression[pos_mask])
            regression_loss = reg_loss.sum() / pos_mask.sum()
        else:
            regression_loss = torch.tensor(0.0).cuda()
        
        # Weighted Sum (Based on config)
        total_loss = classification_loss + regression_loss
        
        loss_dict = {
            'classification_loss': classification_loss.item(),
            'regression_loss': regression_loss.item()
        }
        
        return classification_loss, regression_loss, loss_dict

    def build_targets(self, anchors, annotations):
        """
        Matches anchors to ground truth boxes to create training targets.
        """
        batch_size = len(annotations)
        num_anchors = anchors.shape[1]
        
        cls_targets = torch.zeros((batch_size, num_anchors, self.conf.layer_cfg.num_cls_output)).cuda() - 1 # -1 is ignore
        reg_targets = torch.zeros((batch_size, num_anchors, self.conf.layer_cfg.num_reg_output)).cuda()
        
        pos_mask_batch = torch.zeros((batch_size, num_anchors), dtype=torch.bool).cuda()
        neg_mask_batch = torch.zeros((batch_size, num_anchors), dtype=torch.bool).cuda()

        for i in range(batch_size):
            if len(annotations[i]) == 0:
                continue

            gt_boxes = annotations[i][:, 4:8] # 2D Bboxes [x1, y1, x2, y2]
            
            # Calculate IoU
            ious = calc_iou(anchors[i, :, :4], gt_boxes) # [num_anchors, num_gt]
            
            # Assign Targets
            max_iou, max_idx = torch.max(ious, dim=1)
            
            # Positive Anchors
            pos_mask = max_iou > self.conf.fg_iou_threshold
            pos_mask_batch[i] = pos_mask
            
            if pos_mask.sum() > 0:
                assigned_gt_idx = max_idx[pos_mask]
                
                # Classification Target (1 for car)
                cls_targets[i, pos_mask, 0] = 1 # Assuming 'Car' is index 0
                cls_targets[i, pos_mask, 1:] = 0 # Others are 0
                
                # Regression Target (GT box values)
                # Note: This is a simplified placeholder. Real 3D matching requires transforming 
                # the 3D values (z, w, h, l, theta) into the regression delta format.
                # Assuming 'annotations' contains the pre-encoded regression targets or raw values
                # Standard VisualDet3D usually encodes them here.
                # For now, we pass the raw values to avoid shape mismatch, but for full training
                # ensuring the 'annotations' tensor layout matches 'reg_targets' layout is key.
                gt_values = annotations[i][assigned_gt_idx]
                
                # [x1, y1, x2, y2, cx, cy, z, w, h, l, alpha, ...]
                # This depends on exactly how 'annotations' is collated in your dataset.py
                # Usually it is: [bbox2d(4), bbox3d(7), label(1)]
                # reg_targets should match the 12 outputs of the head.
                
                # Copying whatever is in annotation to match regression shape
                # This logic assumes the dataset loader already prepared regression targets
                # If not, you need the specific encoding function here.
                reg_targets[i, pos_mask] = gt_values[:, :12] 

            # Negative Anchors
            neg_mask = max_iou < self.conf.bg_iou_threshold
            neg_mask_batch[i] = neg_mask
            cls_targets[i, neg_mask, :] = 0
            
        return cls_targets, reg_targets, pos_mask_batch, neg_mask_batch


#from easydict import EasyDict
#from typing import List, Dict, Tuple, Optional
#import numpy as np
#import torch
#import torch.nn as nn
#from visualDet3D.networks.utils.utils import calc_iou
#from visualDet3D.utils.timer import profile
#
#
#class SigmoidFocalLoss(nn.Module):
#    def __init__(self, gamma=0.0, balance_weights=torch.tensor([1.0], dtype=torch.float)):
#        super(SigmoidFocalLoss, self).__init__()
#        self.gamma = gamma
#        self.register_buffer("balance_weights", balance_weights)
#
#    def forward(self, classification:torch.Tensor, 
#                      targets:torch.Tensor, 
#                      gamma:Optional[float]=None, 
#                      balance_weights:Optional[torch.Tensor]=None)->torch.Tensor:
#        """
#            input:
#                classification  :[..., num_classes]  linear output
#                targets         :[..., num_classes] == -1(ignored), 0, 1
#            return:
#                cls_loss        :[..., num_classes]  loss with 0 in trimmed or ignored indexes 
#        """
#        if gamma is None:
#            gamma = self.gamma
#        if balance_weights is None:
#            balance_weights = self.balance_weights
#
#        probs = torch.sigmoid(classification) #[B, N, 1]
#        focal_weight = torch.where(torch.eq(targets, 1.), 1. - probs, probs)
#        focal_weight = torch.pow(focal_weight, gamma)
#
#        bce = -(targets * nn.functional.logsigmoid(classification)) * balance_weights - ((1-targets) * nn.functional.logsigmoid(-classification)) #[B, N, 1]
#        cls_loss = focal_weight * bce
#
#        ## neglect  0.3 < iou < 0.4 anchors
#        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
#
#        ## clamp over confidence and correct ones to prevent overfitting
#        cls_loss = torch.where(torch.lt(cls_loss, 1e-5), torch.zeros(cls_loss.shape).cuda(), cls_loss) #0.02**2 * log(0.98) = 8e-6
#
#        return cls_loss
#
#class SoftmaxFocalLoss(nn.Module):
#    def forward(self, classification:torch.Tensor, 
#                      targets:torch.Tensor, 
#                      gamma:float, 
#                      balance_weights:torch.Tensor)->torch.Tensor:
#        ## Calculate focal loss weights
#        probs = torch.softmax(classification, dim=-1)
#        focal_weight = torch.where(torch.eq(targets, 1.), 1. - probs, probs)
#        focal_weight = torch.pow(focal_weight, gamma)
#
#        bce = -(targets * torch.log_softmax(classification, dim=1))
#
#        cls_loss = focal_weight * bce
#
#        ## neglect  0.3 < iou < 0.4 anchors
#        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
#
#        ## clamp over confidence and correct ones to prevent overfitting
#        cls_loss = torch.where(torch.lt(cls_loss, 1e-5), torch.zeros(cls_loss.shape).cuda(), cls_loss) #0.02**2 * log(0.98) = 8e-6
#        cls_loss = cls_loss * balance_weights
#        return cls_loss
#
#
#class ModifiedSmoothL1Loss(nn.Module):
#    def __init__(self, L1_regression_alpha:float):
#        super(ModifiedSmoothL1Loss, self).__init__()
#        self.alpha = L1_regression_alpha
#
#    def forward(self, normed_targets:torch.Tensor, pos_reg:torch.Tensor):
#        regression_diff = torch.abs(normed_targets - pos_reg) #[K, 12]
#        ## Smoothed-L1 formula:
#        regression_loss = torch.where(
#            torch.le(regression_diff, 1.0 / self.alpha),
#            0.5 * self.alpha * torch.pow(regression_diff, 2),
#            regression_diff - 0.5 / self.alpha
#        )
#        ## clipped to avoid overfitting
#        regression_loss = torch.where(
#           torch.le(regression_diff, 0.01),
#           torch.zeros_like(regression_loss),
#           regression_loss
#        )
#
#        return regression_loss
#
#class IoULoss(nn.Module):
#    """Some Information about IoULoss"""
#    def forward(self, preds:torch.Tensor, targets:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
#        """IoU Loss
#
#        Args:
#            preds (torch.Tensor): [x1, y1, x2, y2] predictions [*, 4]
#            targets (torch.Tensor): [x1, y1, x2, y2] targets [*, 4]
#
#        Returns:
#            torch.Tensor: [-log(iou)] [*]
#        """
#        
#        # overlap
#        lt = torch.max(preds[..., :2], targets[..., :2])
#        rb = torch.min(preds[..., 2:], targets[..., 2:])
#        wh = (rb - lt).clamp(min=0)
#        overlap = wh[..., 0] * wh[..., 1]
#
#        # union
#        ap = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1])
#        ag = (targets[..., 2] - targets[..., 0]) * (targets[..., 3] - targets[..., 1])
#        union = ap + ag - overlap + eps
#
#        # IoU
#        ious = overlap / union
#        ious = torch.clamp(ious, min=eps)
#        return -ious.log()
