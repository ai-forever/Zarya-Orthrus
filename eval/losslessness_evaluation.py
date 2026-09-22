"""
This code calculates various statistics for the Orthrus and Qwen trajectories
presented in the paper https://arxiv.org/abs/2609.15504.
"""

import collections
import json
import math
import os
import numpy as np
import sys
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
import terminaltables
import textdistance
from scipy.stats import pearsonr
from scipy.stats import t
import scipy.stats
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


AUTHORS_CHECKPOINT = "chiennv/Orthrus-Qwen3-1.7B"
MY_ORTHRUS_CHECKPOINT = "/media/inkoziev/corpora/tmp/orthrus/Orthrus-1.7B-final"  # Fix this path


DTYPE = torch.float16  # Options: torch.bfloat16 | torch.float16 | torch.float32
ATTN_IMPLEMENTATION = "eager"   # Options: eager | "sdpa" | "flash_attention_2"


def calc_margin_of_error(measurements) -> float:
    # Step 1: Compute mean
    # mean = np.mean(measurements)

    # Step 2: Compute standard deviation
    std_dev = np.std(measurements, ddof=1)  # ddof=1 for sample standard deviation

    # Step 3: Compute standard error of the mean (SEM)
    sem = std_dev / np.sqrt(len(measurements))

    # Step 4: Get critical t-value (for 95% CI)
    confidence_level = 0.95
    degrees_freedom = len(measurements) - 1
    t_critical = scipy.stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

    # Step 5: Compute margin of error and confidence interval
    margin_of_error = t_critical * sem
    return margin_of_error


def calc_margin_of_error_wilson(measurements, confidence_level=0.95) -> float:
    """Return the half-width of a Wilson score confidence interval
    for a binary proportion.
    """
    measurements = np.asarray(measurements, dtype=np.int64)

    n = len(measurements)
    if n == 0:
        raise ValueError("measurements must not be empty")

    k = np.sum(measurements)
    p = k / n

    z = scipy.stats.norm.ppf((1 + confidence_level) / 2)
    denominator = 1 + z**2 / n

    return (
        z
        * np.sqrt(
            p * (1 - p) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )


def calc_wilson_ci(measurements, confidence_level=0.95):
    """Return the Wilson score confidence interval for a binary proportion."""
    measurements = np.asarray(measurements, dtype=np.int64)

    n = len(measurements)
    if n == 0:
        raise ValueError("measurements must not be empty")

    k = np.sum(measurements)
    p = k / n

    z = scipy.stats.norm.ppf((1 + confidence_level) / 2)

    denominator = 1 + z**2 / n

    center = (p + z**2 / (2 * n)) / denominator

    half_width = (
        z
        * np.sqrt(
            p * (1 - p) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )

    return center - half_width, center + half_width


def mean_ppl_ci(ppls, confidence=0.95):
    """
    Compute arithmetic mean PPL and a Student's t 95% CI.

    Returns:
        mean_ppl: arithmetic mean
        ci_half_width: half-width of the CI
    """

    if not ppls:
        # Empty set is not to be processing.
        return math.nan, math.nan

    ppls = np.asarray(ppls, dtype=float)

    if ppls.ndim != 1 or len(ppls) < 2:
        raise ValueError("ppls must contain at least two values.")

    n = len(ppls)
    mean_ppl = np.mean(ppls)
    std = np.std(ppls, ddof=1)

    se = std / math.sqrt(n)
    critical = t.ppf((1 + confidence) / 2, df=n - 1)

    ci_half_width = critical * se

    return mean_ppl, ci_half_width


ds_fp = "speed_evaluation_dataset.json"
with open(ds_fp) as f:
    domains = json.load(f)

domain_names = sorted(list(set(domains.keys())))


def generate(output_fp: str):
    model_ids = []
    model_ids.append(AUTHORS_CHECKPOINT)
    model_ids.append(MY_ORTHRUS_CHECKPOINT)
    model_ids.append("Qwen/Qwen3-1.7B")

    all_generated_samples = []

    model_args = {"dtype": DTYPE,
                  "device_map": "cuda",
                  "attn_implementation": ATTN_IMPLEMENTATION,
                  "trust_remote_code": True}

    for model_id in model_ids:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_args)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        generation_args = {"max_new_tokens": 128,
                           "do_sample": False,
                           "temperature": 0.0,
                           "pad_token_id": tokenizer.eos_token_id}

        environment_params = dict()
        environment_params["Python version"] = sys.version.split()[0]

        # PyTorch & CUDA
        environment_params["PyTorch version"] = torch.__version__
        environment_params["CUDA version (from torch)"] = torch.version.cuda
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        environment_params["GPU"] = f"{props.name} (compute {props.major}.{props.minor}, {props.total_memory // 1024 ** 3} GB)\n\n"

        # Transformers
        environment_params["Transformers version"] = transformers.__version__

        # Flash Attention (optional)
        try:
            import flash_attn

            environment_params["Flash Attention"] = flash_attn.__version__
        except ImportError:
            environment_params["Flash Attention"] = "not installed"

        environment_params["model_args"] = model_args
        environment_params["generation_args"] = generation_args

        for domain_name, prompts in domains.items():
            for iprompt, prompt in tqdm.tqdm(enumerate(prompts), total=len(prompts), desc=domain_name):
                messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(messages,
                                                          return_tensors="pt",
                                                          add_generation_prompt=True,
                                                          enable_thinking=False).input_ids

                output_ids = model.generate(
                    input_ids=input_ids.to(model.device),
                    **generation_args
                )
                output_ids = output_ids[0].tolist()[input_ids.shape[1]:]
                output_text = tokenizer.decode(output_ids)

                sample = {
                    "model_id": model_id,
                    "attn_implementation": ATTN_IMPLEMENTATION,
                    "model_args": dict(model_args),
                    "generation_args": dict(generation_args),
                    "domain": domain_name,
                    "prompt_index": iprompt,
                    "prompt": prompt,
                    "input_ids": input_ids[0].tolist(),
                    "output_ids": output_ids,
                    "output_text": output_text
                }

                all_generated_samples.append(sample)

                class TorchDtypeEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, torch.dtype):
                            return str(obj)  # e.g., "torch.bfloat16"
                        # Let the base class default method raise the TypeError for other unknown types
                        return super().default(obj)

                with open(output_fp, "w") as f:
                    data = {"environment_params": environment_params,
                            "generations": all_generated_samples}
                    json.dump(data, f, indent=4, cls=TorchDtypeEncoder, ensure_ascii=False)


def first_mismatch_index(list1, list2):
    # Compare elements up to the length of the shorter list
    for i, (elem1, elem2) in enumerate(zip(list1, list2)):
        if elem1 != elem2:
            return i

    # If all common elements match, check if lengths differ
    if len(list1) != len(list2):
        # The first mismatch is the index where the shorter list ends
        return min(len(list1), len(list2))

    # Lists are completely identical
    return -1


def calc_ppl(model, token_ids) -> float:
    t = torch.LongTensor(token_ids).unsqueeze(dim=0).to(model.device)
    with torch.no_grad():
        loss = model(t, labels=t)
    perplexity = math.exp(loss[0].item())
    return perplexity


def calc_conditional_response_ppl(
    model,
    prompt_token_ids,
    response_token_ids,
) -> float:
    """
    Calculate response-conditional perplexity:

        PPL(response | prompt)

    The model receives the complete prompt + response sequence, but
    loss is computed only over response tokens.
    """
    if len(response_token_ids) == 0:
        raise ValueError("response_token_ids must not be empty.")

    input_ids = prompt_token_ids + response_token_ids

    # Ignore prompt tokens when computing the loss.
    labels = (
        [-100] * len(prompt_token_ids)
        + response_token_ids
    )

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    labels = torch.tensor(
        labels,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
        )

    # outputs.loss is the mean cross-entropy over non-masked labels,
    # i.e. response tokens only.
    perplexity = math.exp(outputs.loss.item())

    return perplexity


def process_generations(generations_fp: str, report_fp: str):
    prompt2generations = collections.defaultdict(list)
    prompt2domain = dict()
    prompt2input_ids = dict()

    model_args = {"dtype": torch.bfloat16,
                  "device_map": "cuda",
                  "attn_implementation": "flash_attention_2",
                  "trust_remote_code": True}
    model_id = "Qwen/Qwen3-1.7B"
    qwen_model = AutoModelForCausalLM.from_pretrained(model_id, **model_args)

    with open(generations_fp) as f:
        data = json.load(f)
        for generation in data["generations"]:
            prompt = generation["prompt"]
            prompt2domain[prompt] = generation["domain"]
            prompt2input_ids[prompt] = generation["input_ids"]
            prompt2generations[prompt].append(generation)

    domain_model_mismatch_rate = collections.defaultdict(list)
    domain_model_first_mismatch_index = collections.defaultdict(list)

    model_ppl_levenstein_dist = collections.defaultdict(list)

    model2records = collections.defaultdict(list)

    for prompt, generations in prompt2generations.items():
        def pick_generation(generations, model_id: str):
            for generation in generations:
                if generation["model_id"] == model_id:
                    return generation
            raise RuntimeError()

        qwen_generation = pick_generation(generations, "Qwen/Qwen3-1.7B")
        orthrus1_generation = pick_generation(generations, AUTHORS_CHECKPOINT)
        orthrus2_generation = pick_generation(generations, MY_ORTHRUS_CHECKPOINT)

        qwen_output_ids = qwen_generation["output_ids"]
        orthrus1_output_ids = orthrus1_generation["output_ids"]
        orthrus2_output_ids = orthrus2_generation["output_ids"]

        domain_model_mismatch_rate[(prompt2domain[prompt], orthrus1_generation["model_id"])].append(qwen_output_ids == orthrus1_output_ids)
        domain_model_mismatch_rate[(prompt2domain[prompt], orthrus2_generation["model_id"])].append(qwen_output_ids == orthrus2_output_ids)

        domain_model_first_mismatch_index[(prompt2domain[prompt], orthrus1_generation["model_id"])].append(first_mismatch_index(qwen_output_ids, orthrus1_output_ids))
        domain_model_first_mismatch_index[(prompt2domain[prompt], orthrus2_generation["model_id"])].append(first_mismatch_index(qwen_output_ids, orthrus2_output_ids))

        token_ids = qwen_generation["input_ids"] + qwen_generation["output_ids"]
        #qwen_ppl0 = calc_ppl(qwen_model, token_ids)
        qwen_ppl = calc_conditional_response_ppl(qwen_model, qwen_generation["input_ids"], qwen_generation["output_ids"])

        # Correlation between prompt+response PPL and Levenstein distance of Qwen and Orthrus generations.
        lev_dist1 = textdistance.levenshtein.distance(qwen_output_ids, orthrus1_output_ids)
        model_ppl_levenstein_dist[orthrus1_generation["model_id"]].append((qwen_ppl, lev_dist1))
        model2records[orthrus1_generation["model_id"]].append({"response_length": len(orthrus1_output_ids),
                                                               "qwen_response_length": len(qwen_generation["output_ids"]),
                                                               "qwen_ppl": qwen_ppl,
                                                               "domain": prompt2domain[prompt],
                                                               "levenstein_distance": lev_dist1})

        lev_dist2 = textdistance.levenshtein.distance(qwen_output_ids, orthrus2_output_ids)
        model_ppl_levenstein_dist[orthrus2_generation["model_id"]].append((qwen_ppl, lev_dist2))
        model2records[orthrus2_generation["model_id"]].append({"qwen_ppl": qwen_ppl,
                                                               "response_length": len(orthrus2_output_ids),
                                                               "qwen_response_length": len(qwen_generation["output_ids"]),
                                                               "domain": prompt2domain[prompt],
                                                               "levenstein_distance": lev_dist2})

    # -----------------------------------------------------------------------------------------

    print("\n\n")

    with open(report_fp, "w") as frep:

        # ----------------------------------------------------------------------
        # Сводная таблица, чтобы показать долю траекторий Orthrus'а, совпадающих
        # с траекториями Qwen'а, без разбивки на домены.
        # ----------------------------------------------------------------------
        table = [["Model", "No.trajectories", "Sequence Match Rate", "Mismatching Trajectories Rate"]]
        model2matches = collections.defaultdict(list)
        for model_id in ["chiennv/Orthrus-Qwen3-1.7B", MY_ORTHRUS_CHECKPOINT]:
            for (domain, model_id2), matches in domain_model_mismatch_rate.items():
                if model_id == model_id2:
                    model2matches[model_id].extend(matches)

            matches = model2matches[model_id]
            mismatches = [not x for x in matches]
            table.append([model_id.split("/")[-1],
                          len(matches),
                          "{:.2f} [{:.3f}, {:.3f}]".format(np.mean(matches), calc_wilson_ci(matches)[0], calc_wilson_ci(matches)[1]),
                          "{:.2f} [{:.3f}, {:.3f}]".format(np.mean(mismatches), calc_wilson_ci(mismatches)[0], calc_wilson_ci(mismatches)[1])
                          ])
        print("\n\n** Orthrus-Qwen Trajectory Matching Statistics **")
        print(terminaltables.AsciiTable(table).table)

        frep.write("\n\n** Orthrus-Qwen Trajectory Matching Statistics **\n\n")
        frep.write(terminaltables.AsciiTable(table).table+"\n\n")

        # -----------------------------------------------------------------
        table = [["Model", "Domain", "No.trajectories", "Trajectory length", "Sequence Match Rate", "First Divergence Position", "mean levenstein(Orthrus,Qwen)"]]
        for (domain, model_id), matches in domain_model_mismatch_rate.items():
            # first mismatch index
            fmi_values = domain_model_first_mismatch_index[(domain, model_id)]
            fmi_values = [
                x for x in domain_model_first_mismatch_index[(domain, model_id)]
                if x >= 0  # eliminate exact matching trajectories
            ]

            trajectory_lengths = []

            # Find records for model+domain and calculate the mean levenstein distance
            num_trajectories = 0
            lev_dists = []
            for record in model2records[model_id]:
                if record["domain"] == domain:
                    lev_dists.append(record["levenstein_distance"])
                    trajectory_lengths.append(record["response_length"])
                    num_trajectories += 1

            table.append((model_id.split("/")[-1],
                          domain,
                          num_trajectories,
                          "{:.1f} ± {:.1f}".format(np.mean(trajectory_lengths), calc_margin_of_error(trajectory_lengths)),
                          "{:.2f} ± {:.2f}".format(np.mean(matches), calc_margin_of_error(matches)),
                          "{:.1f} ± {:.1f}".format(np.mean(fmi_values), calc_margin_of_error(fmi_values)),
                          "{:.1f} ± {:.1f}".format(np.mean(lev_dists), calc_margin_of_error(lev_dists))))
        print("\n\n** Detailed statistics of Orthrus trajectories **")
        print(terminaltables.AsciiTable(table).table)

        frep.write("\n\n** Detailed statistics of Orthrus trajectories **\n\n")
        frep.write(terminaltables.AsciiTable(table).table+"\n\n")

        # Краткий вариант табицы с точностью траекторий по доменам, для вставки в squib.

        table = [["Domain", "Authors", "Ours"]]
        for domain in domain_names:
            matches_authors = domain_model_mismatch_rate[domain, AUTHORS_CHECKPOINT]
            matches_ours = domain_model_mismatch_rate[domain, MY_ORTHRUS_CHECKPOINT]

            table.append((domain,
                          "{:.2f} ± {:.2f}".format(np.mean(matches_authors), calc_margin_of_error_wilson(matches_authors)),
                          "{:.2f} ± {:.2f}".format(np.mean(matches_ours), calc_margin_of_error_wilson(matches_ours)),
                        ))
        print("\n\n** Orthrus-Qwen trajectory matching per domain **")
        print(terminaltables.AsciiTable(table).table)

        frep.write("\n\n** Orthrus-Qwen trajectory matching per domain **\n\n")
        frep.write(terminaltables.AsciiTable(table).table+"\n\n")





        # Нарисуем гистограммы PPL для случаев "Orthrus trajectory matching" и "Orthrus trajectory mismatching", то есть когда сгенерированные последовательности
        # токенов полностью совпадают и когда есть хотя бы одно отличие.
        for model, pairs in model_ppl_levenstein_dist.items():
            model_id = model.split("/")[-1]
            ppls1 = [ppl for ppl, lev_dist in pairs if lev_dist==0]  # PPL for exact matching trajectories
            ppls2 = [ppl for ppl, lev_dist in pairs if lev_dist>0]  # PPL for mismatching trajectories

            # Plot both histograms
            plt.hist(ppls1, bins=20, alpha=0.7, color='green', label="Matching trajectory PPL", edgecolor='black')
            plt.hist(ppls2, bins=20, alpha=0.7, color='crimson', label='Diverging trajectory PPL', edgecolor='black')

            plt.xlabel('PPL(response|prompt)')
            plt.ylabel('Frequency')
            plt.legend()
            #plt.title(f'Distribution of PPL(response|prompt)\nfor matching and diverging trajectories of {model_id}')
            plt.savefig(f'PPL and sequence matching for {model_id}.dtype={DTYPE}.attention_implementation={ATTN_IMPLEMENTATION}.png')
            plt.clf()

        table = [["Model", "mean PPL for matching trajectories", "mean PPL for mismatching trajectories"]]
        for model, pairs in model_ppl_levenstein_dist.items():
            ppls1 = [ppl for ppl, lev_dist in pairs if lev_dist==0]  # PPL for exact matching trajectories
            ppls2 = [ppl for ppl, lev_dist in pairs if lev_dist>0]  # PPL for mismatching trajectories

            mean_ppl1, ci1 = mean_ppl_ci(ppls1)
            mean_ppl2, ci2 = mean_ppl_ci(ppls2)

            table.append((model.split("/")[-1], f"{mean_ppl1:.2f} ± {ci1:.2f}", f"{mean_ppl2:.2f} ± {ci2:.2f}"))
        print("\n\n** Mean PPL(response|prompt) for matching and mismatching Orthrus trajectories **")
        print(terminaltables.AsciiTable(table).table)

        # -----------------------------------------------------------------------------------------

        # Построим логистическую регрессию, в которой будут следующие x:
        #
        # x1 = log(Qwen PPL)
        # x2 = orthrus_response_length
        # y = 1 если Orthrus matches (сгенерированные токены совпадают с Qwen), 0 если Orthrus diverges (есть отклонения в токенах)

        # То есть:
        # $$ \boxed{ \text{Losslessness} \sim \log(\mathrm{PPL}) + \text{response length} + \text{domain} } $$

        # Для каждого варианта Ортруса строим отдельную регрессию.
        for model_id, records in model2records.items():
            print(f"\n\nFitting logistic regression for {model_id}...\n")

            df = pd.DataFrame({
                "qwen_ppl": [record["qwen_ppl"] for record in records],
                "response_length": [record["qwen_response_length"] for record in records],
                "domain": [record["domain"] for record in records],
                "match": [(1 if record["levenstein_distance"]==0 else 0) for record in records],
            })

            # Используем логрегрессию из statsmodels, чтобы получить статзначимость для коэффициентов регрессии.

            # Log-transform PPL.
            df["log_ppl"] = np.log(df["qwen_ppl"])

            # Logistic regression:
            #
            # match = 1 -> exact match
            # match = 0 -> divergence
            #
            # C(domain) treats domain as categorical rather than numerical.
            result = smf.logit(
                "match ~ log_ppl + response_length + C(domain)",
                data=df,
            ).fit()

            print(result.summary())

            beta1 = result.params["log_ppl"]
            p_value = result.pvalues["log_ppl"]
            ci_low, ci_high = result.conf_int().loc["log_ppl"]

            print("\n\n== Logistic Regression for {} ==".format(model_id))

            print("beta_1:", beta1)
            print("95% CI:", ci_low, ci_high)
            print("p-value:", p_value)

            # Convert β₁ to an odds ratio
            odds_ratio = np.exp(beta1)
            or_low = np.exp(ci_low)
            or_high = np.exp(ci_high)

            print(f"Odds ratio: {odds_ratio:.3f}")
            print(f"95% CI: [{or_low:.3f}, {or_high:.3f}]")

            # Интерпретация коэффициента beta1:
            if beta1 < 0:
                print("Interpretation: higher Qwen PPL is associated with a lower probability of exact Orthrus-Qwen trajectory matching.")
            elif beta1 > 0:
                print("Interpretation: higher Qwen PPL is associated with a higher probability of exact Orthrus-Qwen trajectory matching.")

        # -----------------------------------------------------------------
        table = [["Model", "Pearson's r", "p-value", "Significance"]]
        for model, pairs in model_ppl_levenstein_dist.items():
            x = [p[0] for p in pairs]
            y = [p[1] for p in pairs]

            # Calculate both the correlation coefficient and the p-value
            r, p_value = pearsonr(x, y)  # type: ignore

            # Optional: Format p-value for better display
            if p_value < 0.001:
                p_display = "< 0.001"
            else:
                p_display = f"{p_value:.4f}"

            # A p-value ≤ 0.05 is a common threshold to declare the correlation statistically significant
            if p_value <= 0.05:
                significance = "the correlation is statistically significant"
            else:
                significance = "the correlation is not statistically significant"

            table.append([model.split("/")[-1], f"{r:.4f}", p_display, significance])

        print("\n\nPearson's coefficient of correlation of PPL(response|prompt) and Levenstein distance between Qwen and Orthrus trajectories")
        print(terminaltables.AsciiTable(table).table)

    return


def report_prompts_stats():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    domain_infos = [["Domain", "No. unique prompts", "Mean prompt length, Qwen3 tokens"]]

    for domain in domain_names:
        prompts = domains[domain]
        lens = []
        for prompt in set(prompts):
            messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages,
                                                      add_generation_prompt=True,
                                                      enable_thinking=False).input_ids
            lens.append(len(input_ids))

        domain_infos.append((domain, len(lens), "{:.1f}".format(np.mean(lens))))

    print("** Domain prompt statistics **")
    print(terminaltables.AsciiTable(domain_infos).table+"\n\n")

if __name__ == "__main__":
    # Construct the .json filename including model configuration parameters.
    generations_fp = f"losslessness_evaluation_generations.dtype={DTYPE}.attn_implementation={ATTN_IMPLEMENTATION}.json"
    report_fp = f"losslessness_evaluation_generations.dtype={DTYPE}.attn_implementation={ATTN_IMPLEMENTATION}.txt"

    # Report the token-level statistics for prompts in evaluation dataset.
    report_prompts_stats()

    if not os.path.exists(generations_fp):
        # Generate trajectories
        generate(generations_fp)

    # Evaluate the generated trajectory properties.
    process_generations(generations_fp, report_fp)
