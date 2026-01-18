"""
    This script contains function snippets for different training settings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from easydict import EasyDict
from visualDet3D.utils.utils import LossLogger
from visualDet3D.utils.utils import compound_annotation
from visualDet3D.networks.utils.registry import PIPELINE_DICT

@PIPELINE_DICT.register_module
def train_mono_detection(data, module:nn.Module,
                     optimizer:optim.Optimizer,
                     writer:SummaryWriter=None, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None,
                     epoch_num:int=None,
                     cfg:EasyDict=EasyDict()):
    optimizer.zero_grad()
    images, P2, labels, bbox2d, bbox_3d, depth, foggy_images = data

    # create compound array of annotation
    max_length = np.max([len(label) for label in labels])
    if max_length == 0:
       return
    annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, cfg.obj_types) #np.arraym, [batch, max_length, 4 + 1 + 7]

    # # Feed to the network
    # classification_loss, regression_loss, l_proposed ,loss_dict = module(
    #         [images.cuda().float().contiguous(), 
    #          images.new(annotation).cuda(),
    #          P2.cuda(),
    #          depth.cuda().contiguous(),
    #          foggy_images.cuda().float().contiguous()]
    #     )

    # classification_loss = classification_loss.mean()
    # regression_loss = regression_loss.mean()

    # if not loss_logger is None:
    #     # Record loss in a average meter
    #     loss_logger.update(loss_dict)
    # del loss_dict


    # Feed to the network
    # Expecting 5 return values now: cls, reg, shortcut_dict, total_diff_loss, det_loss_dict
    # classification_loss, regression_loss, shortcut_loss_dict, l_proposed, loss_dict = module(
    #         [images.cuda().float().contiguous(), 
    #          images.new(annotation).cuda(),
    #          P2.cuda(),
    #          depth.cuda().contiguous(),
    #          foggy_images.cuda().float().contiguous()]
    #     )
    
    _, _, classification_loss, regression_loss, shortcut_loss_dict, l_proposed, loss_dict = module(
            [images.cuda().float().contiguous(), 
             images.new(annotation).cuda(), 
             P2.cuda(), 
             depth.cuda().contiguous(), 
             foggy_images.cuda().float().contiguous()]
        )

    classification_loss = classification_loss.mean()
    regression_loss = regression_loss.mean()
    l_proposed = l_proposed.mean() # Ensure scalar

    if not loss_logger is None:
        # 1. Record standard detection losses (cls, reg, etc.)
        loss_logger.update(loss_dict)
        
        # 2. Record our new Shortcut losses for Tensorboard/Console
        # We use .item() to detach from GPU graph and save memory
        loss_logger.update(dict(
            proposed_loss = l_proposed.item(),
            l_grounding   = shortcut_loss_dict['l_grounding'].item(),
            l_consistency = shortcut_loss_dict['l_consistency'].item()
        ))
        
    del loss_dict, shortcut_loss_dict

    if not optimizer is None:
        loss = classification_loss + regression_loss + l_proposed

    if bool(loss == 0):
        del loss, loss_dict
        return
    loss.backward()
    # clip loss norm
    torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.clipped_gradient_norm)

    optimizer.step()
    optimizer.zero_grad()
