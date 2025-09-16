import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import torch
import transformers
import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils
import gptq_utils
from flatquant import obr_utils
from flatquant.ppl_utils import eval_ppl
import lm_eval
from lm_eval.models.huggingface import HFLM


def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)

    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)

    # get calibration data
    trainloader = data_utils.get_loaders(
        args, args.cali_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False
    )
    logger.info("Finished loading training data.")

    if args.quantize:
        model = apply_flatquant_to_model(args, model)
        logger.info("Finished applying FlatQuant to model.")
        if args.resume:
            flat_utils.load_flat_parameters(args, model)
        elif args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (args.cali_trans or args.add_diag or args.lwc or args.lac):
            train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger)
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model)
        flat_utils.reparameterize_model(model)
        logger.info("Finished reparameterize model.")

    if args.w_bits < 16 and not args.load_qmodel_path:
        save_dict = {}
        quantizers = obr_utils.obr_fwrd(model, trainloader, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers


    # Load Quantized Rotated Model
    ##############################
    if args.load_qmodel_path:
        print("Load quantized model from ", args.load_qmodel_path)
        save_dict = model_utils.load_state_dict_from_HF(args.load_qmodel_path)
        msg = model.load_state_dict(save_dict, strict=False)
        print(msg)

    if args.save_qmodel_path:
        os.makedirs(os.path.dirname(args.save_qmodel_path), exist_ok=True)
        save_dict = model.state_dict()
        torch.save(save_dict, args.save_qmodel_path)
    ##############################

    model_utils.check_sparsity(model)

    utils.distribute_model(model)


    # Evaluating PPL
    print("*" * 30)
    ###############################################################################################################
    eval_batch_size = 1
    model.eval()

    if args.ppl_eval:
        ppl_test = eval_ppl(None, model, tokenizer, device=None)
        print(f"wikitext perplexity using wanda codebase {ppl_test}")
        print('=' * 12)


    # ============== Run Zeroshot Eval ================
    if args.lm_eval:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)
        with torch.no_grad():
            print("Evaluating Zero-Shot!")
            zero_shot_tasks = ['piqa', "boolq", "hellaswag", "arc_easy", 'arc_challenge', "winogrande"]

            ### LM Eval Harness ###
            zs_results = lm_eval.simple_evaluate(hflm, tasks=zero_shot_tasks, num_fewshot=0, batch_size=eval_batch_size)['results']
            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in zs_results.items()}
            print('+' * 10)
            for k, v in metric_vals.items():
                print("Task Name: " + k + " Task Score: ", v)
            print('+' * 10)



if __name__ == '__main__':
    main()