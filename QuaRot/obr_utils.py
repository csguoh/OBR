import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import logging
from prune_utils import wanda_pruning, sparseGPT_pruning, magnitude_pruning,random_pruning


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



def obr_rtn_quantization(W, quantizer):
    q = quantizer.quantize(W)
    return q


def obr_gptq_quantization(W, quantizer, XTX, blocksize, W_mask, obr_alpha):
    columns = W.shape[1]
    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)
    H = XTX
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    for i1 in range(0, columns, blocksize):
        if i1 < obr_alpha * columns:
            continue
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            col_idx = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i]
            w = w
            q = quantizer.quantize(w.unsqueeze(1)).flatten()
            q = q * W_mask[:, col_idx]

            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    q2 = Q[:, int(0.5 * W.shape[1]):]
    return q2





class OBR_Wrapper:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0


    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(inp.float(), p=2, dim=-1) ** 2 / self.nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


    def baseline_with_sparsegpt_gptq_obs(self,percdamp=.01, sparsity_ratio=0.5,prune_n=0, prune_m = 0, blocksize=128,):
        # print('Baseline for Comparison with SparseGPT+GPTQ')
        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        Losses = torch.zeros(self.rows, device=self.dev)

        try:
            H_tmp = H.clone()
            damp = percdamp * torch.mean(torch.diag(H_tmp))
            diag = torch.arange(self.columns, device=self.dev)
            H_tmp[diag, diag] += damp
            H_tmp = torch.linalg.cholesky(H_tmp)
            H_tmp = torch.cholesky_inverse(H_tmp)
            H_tmp = torch.linalg.cholesky(H_tmp, upper=True)
            Hinv = H_tmp


        except torch._C._LinAlgError:
            print("The matrix is not postive-definite, try a larger percdamp!")
            percdamp = 0.1
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity_ratio)]
                mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                q = self.quantizer.quantize(q.unsqueeze(1)).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        ##############
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')



    def optimal_brain_restoration(self, percdamp=.01,sparsity_ratio = 0.5, prune_n=0, prune_m = 0, obr_rtn=False,obr_alpha=0.5, blocksize=128,):
        # print('Optimal Brain Restoration')
        W = self.layer.weight.data.clone()
        W = W.float()
        if not self.quantizer.ready():
            self.quantizer.find_params(W)
        XTX = self.H
        dead = torch.diag(XTX) == 0
        XTX[dead, dead] = 1
        W[:, dead] = 0
        del self.H
        damp = percdamp * torch.mean(torch.diag(XTX))
        XTX += damp * torch.eye(XTX.size(0), device=XTX.device)
        W_mask = wanda_pruning(weight=W, scaler_row=self.scaler_row, sparsity_ratio=sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
        W_bar = W * W_mask
        torch.cuda.synchronize()
        ##############

        #################
        Delta = torch.zeros_like(W)

        for c in range(self.rows):
            mask_row = W_mask[c]
            I = torch.nonzero(mask_row, as_tuple=False).squeeze(1)
            Z = torch.nonzero(~mask_row.bool(), as_tuple=False).squeeze(1)
            if I.numel() == 0 or Z.numel() == 0:
                continue

            H_II = XTX[I][:, I]
            H_IZ = XTX[I][:, Z]
            W_z = W[c, Z]
            b = H_IZ.matmul(W_z)

            L = torch.linalg.cholesky(H_II)
            delta_w = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)

            w_prime = W_bar[c,I].clone()
            w_prime =  w_prime + delta_w

            quantized = self.quantizer.quantize_one_row(w_prime, c)
            e_full =  w_prime - quantized
            partation_point = torch.searchsorted(I, int(W.shape[1]*obr_alpha), right=False)
            idx_1 =  torch.arange(0, partation_point)
            idx_2 =  torch.arange(partation_point,I.shape[0])
            e_full = e_full[idx_1]
            HII_1 = H_II[idx_2][:, idx_1]
            HII_2 = H_II[idx_2][:, idx_2]
            b2 = HII_1.matmul(e_full)
            L2 = torch.linalg.cholesky(HII_2)
            delta_w1 = torch.cholesky_solve(b2.unsqueeze(-1), L2).squeeze(-1)

            delta_w[idx_2] += delta_w1
            Delta[c, I] = delta_w


        W = W_bar + Delta
        torch.cuda.synchronize()
        #####################################
        if obr_rtn:
            q = obr_rtn_quantization(W,self.quantizer)
        #########################################
        else:
            q1 = self.quantizer.quantize(W[:, :int(obr_alpha*W.shape[1])])
            q2 = obr_gptq_quantization(W, self.quantizer, XTX, blocksize, W_mask,obr_alpha)
            q = torch.cat((q1, q2), dim=1)

        self.layer.weight.data = q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')


    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


        
@torch.no_grad()
def obr_fwrd(model, dataloader, dev, args):
    logging.info('-----Join Quantization & Sparsification Start!-----\n')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
                ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
                ['self_attn.o_proj.module'],
                ['mlp.up_proj.module', 'mlp.gate_proj.module'],
                ['mlp.down_proj.module']
            ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = OBR_Wrapper(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                # gptq[name].baseline_with_sparsegpt_gptq_obs(percdamp=args.percdamp, sparsity_ratio=args.sparsity_ratio, prune_n=args.prune_n, prune_m = args.prune_m)
                gptq[name].optimal_brain_restoration(percdamp=args.percdamp,
                                                     sparsity_ratio=args.sparsity_ratio,
                                                     prune_n=args.prune_n, prune_m = args.prune_m,
                                                     obr_alpha=args.obr_alpha,
                                                     obr_rtn=args.obr_rtn)
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        #break


    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----Join Quantization & Sparsification Done!-----\n')
    return quantizers

