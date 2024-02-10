import torch
from tqdm import tqdm
import time
import numpy as np
import os
import logging


def compute_loss(criterion, y_pred, y_gt, loss_mask):
    """
    Args:
        criterion: MSELoss(reduction='none')
        y_pred, y_gt: (bs, seq_len, D)
        loss_mask: (bs, seq_len, 1)
    Returns:
        loss: Tensor of shape []
    """
    # (bs, seq_len, D)
    loss = criterion(y_pred, y_gt)
    loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask)
    return loss


def evaluate(args, loader, model, criterion, dataset_type):
    logging.info(
        "Eval time:".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        )
    )

    model.eval()
    eval_loss = 0.0

    pred_res = []
    with torch.no_grad():
        for idxs, y_gt, y_mask in tqdm(loader):
            # y_gt: (bs, seq, sp_dim)
            # y_masks: (bs, seq, 1)
            # y_pred: (bs, seq, output_dim)
            # masks: (bs, seq, 1)

            y_pred, masks = model(idxs, loader.dataset)
            loss = compute_loss(criterion, y_pred, y_gt, masks * y_mask)

            eval_loss += loss.item()
            pred_res.append(y_pred.cpu().detach().numpy())

        eval_loss /= len(loader)

    # (dataset_sz, seq, sp_dim)
    pred_res = np.concatenate(pred_res)
    pred_file = os.path.join(
        args.save, "{}_{}.npy".format(dataset_type, args.current_epoch)
    )
    return eval_loss, pred_res, pred_file
