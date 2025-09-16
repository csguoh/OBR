import torch
import transformers
import logging
from flatquant.utils import skip
from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31
import os
import json
import gptq_utils



def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        layers = model.model.layers
    except:
        layers = model.model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = gptq_utils.find_qlayers(layer)

        for name in subset:
            W = subset[name].weight.data
            cur_zeros = (W==0).sum().item()
            cur_total = W.numel()

            count += cur_zeros
            total_params += cur_total

            print(f"layer {i} name {name} {W.shape} sparsity {float(cur_zeros)/cur_total}")

    print(f"total number of params {total_params}")
    model.config.use_cache = use_cache
    return float(count)/total_params




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



def skip_initialization():
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def get_llama(model_name, hf_token):
    skip_initialization()
    config = transformers.LlamaConfig.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = transformers.LlamaForCausalLM.from_pretrained(model_name,
                                                          torch_dtype='auto',
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')
    return model, apply_flatquant_to_llama


def get_llama_31(model_name, hf_token):
    skip_initialization()
    config = transformers.LlamaConfig.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = transformers.LlamaForCausalLM.from_pretrained(model_name,
                                                          torch_dtype='auto',
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')
    return model, apply_flatquant_to_llama_31


def get_qwen2(model_name, hf_token):
    skip_initialization()
    try:
        from transformers import Qwen2ForCausalLM
    except ImportError:
        logging.error("Qwen2 model is not available in this version of 'transformers'. Please update the library.")
        raise ImportError("Qwen2 model is not available. Ensure you're using a compatible version of the 'transformers' library.")

    config = transformers.Qwen2Config.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = Qwen2ForCausalLM.from_pretrained(model_name,
                                                          torch_dtype='auto',
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')

    from flatquant.model_tools.qwen_utils import apply_flatquant_to_qwen
    return model, apply_flatquant_to_qwen


def get_opt(model_name):
    skip_initialization()
    model = transformers.OPTForCausalLM.from_pretrained(model_name,
                                                        torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')
    raise NotImplementedError("Post-processing for OPT model is not implemented yet.")


# Unified model loading function
def get_model(model_name, hf_token=None):
    if 'llama-3.1' in model_name.lower():
        return get_llama_31(model_name, hf_token)
    elif 'llama' in model_name:
        return get_llama(model_name, hf_token)
    elif 'qwen' in model_name or 'Qwen' in model_name:
        return get_qwen2(model_name, hf_token)
    else:
        raise ValueError(f'Unknown model {model_name}')

