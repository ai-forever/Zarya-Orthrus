"""
Side-by-side evaluation of Qwen and Orthrus models using lm-evaluation-harness tasks
and common computational environment. The results are presented in the paper https://arxiv.org/abs/2609.15504
"""

import os
import torch
import lm_eval
import lm_eval.models.huggingface
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # required for MBPP

    # Model argument are common for all 3 models.
    model_args = {"dtype": torch.float16, # torch.float16 | torch.bfloat16 | torch.float32
                  "device_map": "cuda",
                  "attn_implementation": "eager",  # options: sdpa | eager | flash_attention_2 | flash_attention_4
                  "trust_remote_code": True}

    model_ids = []
    model_ids.append("chiennv/Orthrus-Qwen3-1.7B")
    model_ids.append("/media/inkoziev/corpora/tmp/orthrus/Orthrus-1.7B-final")  # local filesystem
    #model_ids.append("/home/jovyan/shares/SR008.fs2/ckp/Orthrus/orthrus-exp42/Orthrus-1.7B-final")  # cluster NFS
    model_ids.append("Qwen/Qwen3-1.7B")

    for model_id in model_ids:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_args)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        eval_model = lm_eval.models.huggingface.HFLM(pretrained=model,
                                                     tokenizer=tokenizer,
                                                     batch_size=1  # NB: Orthrus does not support batched generation, so we restrict all models to batch_size=1
                                                     )

        results = lm_eval.simple_evaluate(
            model=eval_model,
            tasks=["humaneval", "gsm8k", "ifeval"],  # only `generate_until`-based tasks are of our interest.
            num_fewshot=0,
            #limit=0.2,  # set limit for debugging only
            confirm_run_unsafe_code=True,
            gen_kwargs={"do_sample": False},  # greedy decoding for all models
        )
        print("** {} RESULTS **".format(model_id.split("/")[-1]))
        table_str = lm_eval.utils.make_table(results)
        print(table_str)
        with open("evaluation_results.{}.txt".format(model_id.split("/")[-1]), "w", encoding="utf-8") as f:
            f.write(table_str)
