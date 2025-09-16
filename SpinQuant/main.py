# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import datetime
from logging import Logger
import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.ptq import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
import lm_eval
from lm_eval.models.huggingface import HFLM
from ppl_utils import eval_ppl



def train() -> None:
    model_args, training_args, ptq_args = process_args_ptq()
    config = transformers.AutoConfig.from_pretrained(model_args.input_model, token=model_args.access_token)
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model = ptq_model(ptq_args, model, model_args)
    utils.check_sparsity(model)
    model.seqlen = training_args.model_max_length
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    print("Complete tokenizer loading...")
    model.config.use_cache = False

    eval_batch_size = 1
    model.eval()
    utils.distribute_model(model)

    # ============== Run WikiText2 PPL Eval ================
    if ptq_args.ppl_eval:
        ppl_test = eval_ppl(None, model, tokenizer, device=None)
        print(f"wikitext perplexity using wanda codebase {ppl_test}")
        print('=' * 12)

    # ============== Run Zeroshot Eval ================
    if ptq_args.lm_eval:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)
        with torch.no_grad():
            print("Evaluating Zero-Shot!")
            zero_shot_tasks = ['piqa',"boolq","hellaswag", "arc_easy", 'arc_challenge', "winogrande"]

            ### LM Eval Harness ###
            zs_results = lm_eval.simple_evaluate(hflm, tasks=zero_shot_tasks, num_fewshot=0, batch_size=eval_batch_size)['results']

            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in zs_results.items()}

            print('+' * 10)
            for k, v in metric_vals.items():
                print("Task Name: " + k + " Task Score: ", v)
            print('+' * 10)



if __name__ == "__main__":
    train()
