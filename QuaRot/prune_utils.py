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


def random_pruning(weight, sparsity_ratio):
    num_cols = weight.shape[1]
    prune_count = int(num_cols * sparsity_ratio)

    # 每一行随机选择需要剪掉的列索引
    random_indices = torch.stack([
        torch.randperm(num_cols)[:prune_count]
        for _ in range(weight.shape[0])
    ], dim=0)

    # mask: 1表示保留，0表示剪掉
    mask = torch.ones_like(weight)
    mask.scatter_(1, random_indices, 0.0)
    return mask


def magnitude_pruning(weight, sparsity_ratio):
    W_metric = torch.abs(weight.data)
    sort_res = torch.sort(W_metric, dim=-1, stable=True)
    # unstructured pruning
    indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
    W_mask = torch.zeros_like(W_metric)
    W_mask.scatter_(1, indices, 1.0)
    return (1-W_mask).to(weight)



def sparseGPT_pruning(weight, sparsity_ratio, XTX):
    H = torch.linalg.cholesky(XTX)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    W_metric = weight ** 2 / (torch.diag(Hinv).reshape((1, -1))) ** 2 # W^2 / diag(H^{-1})
    sort_res = torch.sort(W_metric, dim=-1, stable=True)
    indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
    W_mask = torch.zeros_like(W_metric)
    W_mask.scatter_(1, indices, 1.0)
    return (1-W_mask).to(weight)
