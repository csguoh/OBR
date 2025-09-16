import torch
import torch.nn.functional as F
import torch.nn as nn


def wanda_pruning(weight, scaler_row, sparsity_ratio,prune_n=0,prune_m=0):

    weight = torch.abs(weight.data)
    activation = torch.sqrt(scaler_row.reshape((1, -1)))

    W_metric = weight * activation
    W_mask = (torch.zeros_like(W_metric) == 1)
    if prune_n != 0:
        # structured n:m sparsity
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
        W_mask.scatter_(1, indices, True)

    return (~W_mask).to(weight)
