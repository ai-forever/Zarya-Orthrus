import time
import json
import sys
import os
import transformers
import torch
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import terminaltables
import torch.nn.functional as F
from transformers.trainer_pt_utils import get_model_param_count
import tqdm
import scipy
import scipy.stats
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import lm_eval


num_new_tokens = 256


# Подготовленный датасет с промптами по тематическим доменам, необходимый для консистентного
# сравнения эффективности разных экспериментальных моделей.
ds_fp = "speed_evaluation_dataset.json"
with open(ds_fp) as f:
    domains = json.load(f)

domain_names = sorted(domains.keys())

# Чекпы для запуска замеров.
models = [
    # 0.6B
    "ai-forever/ZaryaOrthrus-0.6B",

    # 1.7B
    "ai-forever/ZaryaOrthrus-1.7B",

    "chiennv/Orthrus-Qwen3-1.7B",
    #"chiennv/Orthrus-Qwen3-4B",
    #"chiennv/Orthrus-Qwen3-8B"
]


def calc_margin_of_error(measurements, confidence_level: float = 0.95) -> float:
    # Step 1: Compute mean
    # mean = np.mean(measurements)

    # Step 2: Compute standard deviation
    std_dev = np.std(measurements, ddof=1)  # ddof=1 for sample standard deviation

    # Step 3: Compute standard error of the mean (SEM)
    sem = std_dev / np.sqrt(len(measurements))

    # Step 4: Get critical t-value (for 95% CI)
    degrees_freedom = len(measurements) - 1
    t_critical = scipy.stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

    # Step 5: Compute margin of error and confidence interval
    margin_of_error = t_critical * sem
    return margin_of_error


class LmEvalAdapter(lm_eval.api.model.LM):
    def __init__(self, model_name, model, tokenizer):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.speed_scores = []

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError()

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, gen_kwargs).
            context: str
                Context string
            gen_kwargs: dict
                A dictionary of keyword arguments to pass to the generation function e.g. top_k, until, etc.
        :return: list[str]
            A list of model generated continuations.
            continuation: str
                The generated continuation.
        """

        res = []
        for request in tqdm.tqdm(requests, desc=self.model_name, disable=disable_tqdm):
            context, gen_kwargs = request.args
            messages = [{"role": "system", "content": ""}, {"role": "user", "content": context}]
            input_ids = self.tokenizer.apply_chat_template(messages,
                                                        return_tensors="pt",
                                                        add_generation_prompt=True,
                                                        enable_thinking=False).input_ids

            max_new_tokens2 = gen_kwargs.get("max_gen_toks", num_new_tokens)

            if "until" in gen_kwargs:
                del gen_kwargs["until"]

            if "max_gen_toks" in gen_kwargs:
                del gen_kwargs["max_gen_toks"]

            t1 = time.time()
            output_ids = self.model.generate(
                input_ids=input_ids.to(self.model.device),
                max_new_tokens=max_new_tokens2,
                **gen_kwargs
            )
            t2 = time.time()
            response_tokens = output_ids[0][input_ids.shape[1]:]
            self.speed_scores.append(len(response_tokens) / (t2-t1))

            generated_text = self.tokenizer.decode(response_tokens)
            res.append(generated_text)

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError()
        res = []

        for _ in tqdm.tqdm(requests, disable=disable_tqdm):
            res.append(-random.random())

        return res


def eval_orthrus_lmeval_wallclock_speed(model_name, model, tokenizer):
    result_data = []
    trainable_params = get_model_param_count(model, trainable_only=True)

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    eval_model = LmEvalAdapter(model_name=model_name, model=model, tokenizer=tokenizer)

    tasks = ["mbpp", "gsm8k", "wmt16"]

    for task in tasks:
        results = lm_eval.simple_evaluate(
            model=eval_model,
            tasks=[task],
            num_fewshot=0,
            max_batch_size=1,
            limit=0.10,
            confirm_run_unsafe_code=True,

            # # Additional parameters for pass@k
            # gen_kwargs={
            #     "do_sample": True,
            #     "temperature": 0.8,
            #     "top_p": 0.95,
            # }
        )

        tokens_per_second = np.mean(eval_model.speed_scores)
        merr = calc_margin_of_error(eval_model.speed_scores)

        result_data.append((model_name,
                            trainable_params,
                            task,
                            tokens_per_second,
                            merr))

        print("lm_eval wallclock task={} speed: {:.2f} tok/sec".format(task, tokens_per_second))

    return result_data, tasks


def eval_orthrus_wallclock_speed(model_name, model, tokenizer):
    """Замер wall clock скорости (ток/сек)"""

    result_data = []

    trainable_params = get_model_param_count(model, trainable_only=True)

    for domain, prompts in domains.items():
        # НАЧАЛО ОТЛАДКИ
        #prompts = prompts[:10]
        # КОНЕЦ ОТЛАДКИ

        tpsx = []
        for prompt in tqdm.tqdm(prompts, desc=f"{model_name} {domain}"):

            messages = [{"role": "user", "content": prompt}, ]
            input_ids = tokenizer.apply_chat_template(messages,
                                                      return_tensors="pt",
                                                      add_generation_prompt=True,
                                                      enable_thinking=False).input_ids

            start_time = time.time()

            output_ids = model.generate(
                input_ids=input_ids.to(model.device),
                max_new_tokens=num_new_tokens,
                use_diffusion_mode=True
            )

            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time

            output_tokens = output_ids[0, input_ids.shape[1]:].tolist()
            if tokenizer.eos_token_id in output_tokens:
                output_tokens = output_tokens[:output_tokens.index(tokenizer.eos_token_id)+1]
            num_real_tokens = len(output_tokens)

            tokens_per_second = num_real_tokens / elapsed_time
            tpsx.append(tokens_per_second)

        tokens_per_second = np.mean(tpsx)
        merr = calc_margin_of_error(tpsx)

        result_data.append((model_name,
                            trainable_params,
                            domain,
                            tokens_per_second,
                            merr))

    return result_data


@torch.inference_mode()
def my_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = None,
        max_length: int = None,
        temperature: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.8,
        eos_token_id: Optional[int] = None,
        use_diffusion_mode: bool = True,
        **kwargs,
) -> float:
    eos_token_id = eos_token_id or getattr(self.config, "eos_token_id", None)

    device = input_ids.device
    num_input_tokens = input_ids.shape[1]
    max_length = max_length or (num_input_tokens + max_new_tokens)
    block_size = self.config.block_size
    mask_token_id = self.config.mask_token_id
    past_key_values = DynamicCache(config=self.config)

    output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=device)
    output_ids[:, :num_input_tokens] = input_ids

    # Здесь будем накапливать список accepted lengths.
    generation_hops = []

    # Для расчета tokens per forward
    forward_counter = 0
    tokens_generated = 0

    def sample(logits: torch.Tensor):
        if temperature < 1e-5:
            return logits.argmax(dim=-1), None

        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., [-1]]] = -float('Inf')
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            logits[sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1]), probs

    # Initial Pass
    position_ids = torch.arange(num_input_tokens, device=device).unsqueeze(0)
    outputs = self(input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values)
    forward_counter += 1
    tokens_generated += 1

    start_idx = num_input_tokens
    next_token, _ = sample(outputs.logits[:, -1, :])
    output_ids[:, start_idx] = next_token

    while start_idx < max_length - 1:
        diff_len = min(block_size, max_length - start_idx)
        diff_block_ids = torch.full((1, diff_len), mask_token_id, dtype=torch.long, device=device)
        diff_block_ids[:, 0] = output_ids[:, start_idx]
        diff_position_ids = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)

        # Diffusion Path
        diff_outputs = self(
            input_ids=diff_block_ids,
            position_ids=diff_position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            is_diffusion_pass=True,
            ar_seq_len=start_idx,
        )

        if diff_len > 1:
            diff_tokens, diff_probs = sample(diff_outputs.logits[:, :-1, :])
        else:
            diff_tokens, diff_probs = torch.empty((1, 0), dtype=torch.long, device=device), None

        proposed_block = torch.cat([output_ids[:, start_idx:start_idx + 1], diff_tokens], dim=1)

        # Autoregressive Path for Intra-model Consistency
        ar_outputs = self(
            input_ids=proposed_block,
            position_ids=diff_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            is_diffusion_pass=False,
        )
        ar_tokens, ar_probs = sample(ar_outputs.logits)
        forward_counter += 1

        acceptance_len = 0
        if temperature < 1e-5:
            matches = (diff_tokens == ar_tokens[:, :-1])
            acceptance_len = matches.cumprod(dim=1).sum(dim=1)[0].item()
            next_token = ar_tokens[:, acceptance_len]
        else:
            for i in range(diff_tokens.shape[1]):
                q_prob = diff_probs[0, i, diff_tokens[0, i]]
                p_prob = ar_probs[0, i, diff_tokens[0, i]]
                if torch.rand(1, device=device).item() < min(1.0, (p_prob / max(q_prob, 1e-8)).item()):
                    acceptance_len += 1
                else:
                    break

            p_dist = ar_probs[0, acceptance_len]
            if acceptance_len < diff_tokens.shape[1]:
                residual = torch.clamp(p_dist - diff_probs[0, acceptance_len], min=0.0)
                residual_sum = residual.sum()
                next_token = torch.multinomial(residual / residual_sum if residual_sum > 1e-5 else p_dist, 1)
            else:
                next_token = torch.multinomial(p_dist, 1)

        end_idx = start_idx + acceptance_len + 1
        accepted_block = proposed_block[:, :acceptance_len + 1]
        generation_hops.append(acceptance_len)
        tokens_generated += acceptance_len

        eos_positions = (accepted_block == eos_token_id).nonzero()
        if len(eos_positions) > 0:
            eos_offset = eos_positions[0, -1].item()
            output_ids[:, start_idx: start_idx + eos_offset + 1] = accepted_block[:, :eos_offset + 1]
            TPF = tokens_generated / forward_counter
            return np.mean(generation_hops), TPF

        output_ids[:, start_idx:end_idx] = accepted_block

        start_idx = end_idx
        past_key_values.crop(start_idx)

        if start_idx < max_length:
            output_ids[:, start_idx] = next_token

            if next_token.item() == eos_token_id:
                TPF = tokens_generated / forward_counter
                return np.mean(generation_hops), TPF

    TPF = tokens_generated / forward_counter
    return np.mean(generation_hops), TPF


def eval_orthrus_drafter_acceptance(model_name, model, tokenizer):
    """Замеры drafter accepted length и tokens per forward"""
    result_data = []

    old_generate = model.generate  # store bound method
    model.generate = my_generate.__get__(model, model.__class__)

    trainable_params = get_model_param_count(model, trainable_only=True)

    for domain, prompts in domains.items():

        # НАЧАЛО ОТЛАДКИ
        #prompts = prompts[:10]
        # КОНЕЦ ОТЛАДКИ

        accepted_length_stat = []
        tpf_stat = []
        for prompt in tqdm.tqdm(prompts, desc=f"{model_name} {domain}"):
            messages = [{"role": "user", "content": prompt}, ]
            input_ids = tokenizer.apply_chat_template(messages,
                                                      return_tensors="pt",
                                                      add_generation_prompt=True,
                                                      enable_thinking=False).input_ids

            mean_acceptance_length, TPF = model.generate(
                input_ids=input_ids.to(model.device),
                max_new_tokens=num_new_tokens,
                use_diffusion_mode=True
            )
            accepted_length_stat.append(mean_acceptance_length)
            tpf_stat.append(TPF)
            torch.cuda.synchronize()

        mean_acceptance = np.mean(accepted_length_stat)
        merr = calc_margin_of_error(accepted_length_stat)

        TPF = np.mean(tpf_stat)
        tpf_err = calc_margin_of_error(tpf_stat)

        print(f"\ndomain: {domain}  mean_acceptance: {mean_acceptance} ± {merr}  TPF: {TPF} ± {tpf_err}\n")

        result_data.append((model_name,
                            trainable_params,
                            domain,
                            mean_acceptance,
                            merr,
                            TPF,
                            tpf_err))

    # restore
    model.generate = old_generate

    return result_data


if __name__ == "__main__":
    result_data = []  # accepted length and Tokens Per Forward
    result_data2 = []  # Tokens Per Second

    for model_id in models:
        print(f"Loading {model_id}")
        model_name = model_id

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="flash_attention_2",  # options: sdpa | eager | flash_attention_4
            trust_remote_code=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        #model_results, lmeval_tasks = eval_orthrus_lmeval_wallclock_speed(model_name, model, tokenizer)
        #result_data2.extend(model_results)

        #model_results = eval_orthrus_wallclock_speed(model_name, model, tokenizer)
        #result_data2.extend(model_results)

        model_results = eval_orthrus_drafter_acceptance(model_name, model, tokenizer)
        result_data.extend(model_results)

        # После обработки каждого чекпоинта переформируем таблицу результатов.

        # Таблица со статистикой по drafter accepted length
        result_table1 = [["Модель",
                         "Ёмкость",
                         ] + domain_names]

        # Таблица со статистикой по Tokens Per Forward
        result_table2 = [["Модель",
                         "Ёмкость",
                         ] + domain_names]

        # Таблица со статистикой по Tokens Per Second
        result_table3 = [["Модель",
                         "Ёмкость",
                         ] + domain_names]

        # # Таблица со статистикой по Tokens Per Second для тасок lm-eval-harness
        # result_table4 = [["Модель",
        #                  "Ёмкость",
        #                  ] + lmeval_tasks]

        models = sorted(list(set((x[0], x[1]) for x in result_data) | set((x[0], x[1]) for x in result_data2)))
        for model, capacity in models:
            scores1 = dict()
            scores2 = dict()
            for row in result_data:
                if row[0] == model and row[1] == capacity:
                    scores1[row[2]] = (row[3], row[4])  # mean accepted length
                    scores2[row[2]] = (row[5], row[6])  # mean Tokens Per Forward

            result_table1.append(
                [model, "{:.1f}B".format(capacity / 1_000_000_000)] +
                ["{:.2f}".format(scores1[domain][0]) + "<sub>±{:.2f}</sub>".format(scores1[domain][1]) for domain in
                 domain_names])

            result_table2.append(
                [model, "{:.1f}B".format(capacity / 1_000_000_000)] +
                ["{:.2f}".format(scores2[domain][0]) + "<sub>±{:.2f}</sub>".format(scores2[domain][1]) for domain in
                 domain_names])

            # scores3 = dict()
            # for row in result_data2:
            #     if row[0] == model and row[1] == capacity:
            #         scores3[row[2]] = (row[3], row[4])

            # result_table3.append(
            #     [model, "{:.1f}B".format(capacity / 1_000_000_000)] +
            #     ["{:.2f}".format(scores3[domain][0]) + "<sub>±{:.2f}</sub>".format(scores3[domain][1]) for domain in domain_names])

            # result_table4.append(
            #     [model, "{:.1f}B".format(capacity / 1_000_000_000)] +
            #     ["{:.2f}".format(scores3[task][0]) + "<sub>±{:.2f}</sub>".format(scores3[task][1]) for task in lmeval_tasks])

        with open("orthrus_speed_evaluation.md", "w") as f:
            f.write("## Environment\n\n")

            f.write(f"Python version: {sys.version.split()[0]}\n\n")

            # PyTorch & CUDA
            f.write(f"PyTorch version: {torch.__version__}\n\n")
            cuda_avail = torch.cuda.is_available()
            f.write(f"CUDA available: {cuda_avail}\n\n")
            if cuda_avail:
                f.write(f"CUDA version (from torch): {torch.version.cuda}\n\n")
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                f.write(
                    f"GPU: {props.name} (compute {props.major}.{props.minor}, {props.total_memory // 1024 ** 3} GB)\n\n")

            # Transformers
            f.write(f"Transformers version: {transformers.__version__}\n\n")

            # Flash Attention (optional)
            try:
                import flash_attn

                f.write(f"Flash Attention version: {flash_attn.__version__}\n\n")
            except ImportError:
                f.write("Flash Attention: not installed\n\n")

            f.write(f"Параметры генерации: max_new_tokens={num_new_tokens}:\n\n")

            f.write(f"\n\n## Drafter evaluation\n\n")
            f.write("### Drafter Accepted Length\n\n")
            f.write(terminaltables.GithubFlavoredMarkdownTable(result_table1).table + "\n\n")

            f.write("### Tokens Per Forward\n\n")
            f.write(terminaltables.GithubFlavoredMarkdownTable(result_table2).table + "\n\n")

            # f.write("### Tokens Per Second\n\n")
            # f.write(terminaltables.GithubFlavoredMarkdownTable(result_table3).table + "\n\n")

            # f.write("### Tokens Per Second for lm-evaluation-harness tasks\n\n")
            # f.write(terminaltables.GithubFlavoredMarkdownTable(result_table4).table + "\n\n")

        with open("orthrus_speed_evaluation.json", "w") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

        del model
        del tokenizer

    print("\nTokens Per Forward:")
    print(terminaltables.GithubFlavoredMarkdownTable(result_table2).table)

    # print("\nTokens Per Second:")
    # print(terminaltables.GithubFlavoredMarkdownTable(result_table3).table)
