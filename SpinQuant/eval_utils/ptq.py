# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import transformers
import os
from eval_utils import gptq_utils, rotation_utils,obr_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils
from utils.convert_to_executorch import sanitize_checkpoint_from_spinquant, write_model_llama



def load_state_dict_from_HF(local_dir):
    bin_file = os.path.join(local_dir, "pytorch_model.bin")
    safet_file = os.path.join(local_dir, "model.safetensors")
    index_json = os.path.join(local_dir, "pytorch_model.bin.index.json")

    if os.path.exists(bin_file):
        print(f"Loading single-bin state_dict: {bin_file}")
        return torch.load(bin_file, map_location="cpu")

    if os.path.exists(safet_file):
        print(f"Loading safetensors state_dict: {safet_file}")
        from safetensors.torch import load_file
        return load_file(safet_file, device="cpu")

    if os.path.exists(index_json):
        # Sharded .bin case
        print(f"Loading sharded state_dict via index: {index_json}")
        with open(index_json, "r") as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]  # param_name -> shard filename
        state = {}
        loaded = {}
        for shard in set(weight_map.values()):
            shard_path = os.path.join(local_dir, shard)
            print(f"  - merging shard: {shard_path}")
            sd = torch.load(shard_path, map_location="cpu")
            state.update(sd)
            loaded[shard] = True
        return state

    raise FileNotFoundError(f"No state dict found in {local_dir}")



def ptq_model(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()

    # Rotate the weights
    if args.rotate:
        fuse_norm_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.w_bits < 16 and not args.load_qmodel_path:
        save_dict = {}
        trainloader = data_utils.get_wikitext2(
            nsamples=args.nsamples,
            seed=args.seed,
            model=model_args.input_model,
            seqlen=2048,
            eval_mode=False,)
        quantizers = obr_utils.obr_fwrd(model, trainloader, "cuda", args)
        save_dict["w_quantizers"] = quantizers

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name:
                layer_groupsize = head_dim

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name:  # Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
            }
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )

    if args.load_qmodel_path:
        assert args.rotate, "Model should be rotated to load a quantized model!"
        assert (
            not args.save_qmodel_path
        ), "Cannot save a quantized model if it is already loaded!"
        print("Load quantized model from ", args.load_qmodel_path)
        save_dict = load_state_dict_from_HF(args.load_qmodel_path)
        msg = model.load_state_dict(save_dict, strict=False)
        print(msg)

    if args.save_qmodel_path:
        os.makedirs(os.path.dirname(args.save_qmodel_path), exist_ok=True)
        save_dict = model.state_dict()
        torch.save(save_dict, args.save_qmodel_path)


    return model
