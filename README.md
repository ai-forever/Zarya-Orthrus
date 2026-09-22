# Orthrus: Training, Evaluation, and Benchmarking

This repository contains the code and distilled datasets used to train, evaluate,
and benchmark independent implementations of **Orthrus** at the 0.6B and 1.7B
model capacities.

## About Orthrus

[Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion](https://arxiv.org/abs/2605.12825)
introduces a speculative decoding architecture that combines an autoregressive
model with a diffusion-based parallel token generation mechanism. This repository
contains an independent implementation and the associated data distillation,
training, evaluation, and benchmarking code.

## Repository Structure

```text
.
├── eval
│   ├── eval_speed_orthrus.py
│   ├── evaluate-orthrus-via-lm_eval.py
│   ├── losslessness_evaluation.py
│   └── speed_evaluation_dataset.json
├── train
│   ├── configuration.py
│   ├── fp32_logits.py
│   ├── model.py
│   ├── train_orthrus_fused_v40.py
│   └── train_orthrus_fused_v42.py
└── README.md
```

## Datasets

Training datasets are available at:

- [DistillationDataset.Qwen3-0.6B.v29](https://huggingface.co/datasets/DistillationDataset.Qwen3-0.6B.v29) — instructive samples for Orthrus based
  on Qwen3-0.6B.
- [DistillationDataset.Qwen3-1.7B.v29](https://huggingface.co/datasets/DistillationDataset.Qwen3-1.7B.v29) — instructive samples for Orthrus based
  on Qwen3-1.7B.

The datasets are divided into `train` and `test` splits and are stored as JSONL
files, including sharded versions.

The data preparation procedure and the distillation setup will be described in
the technical report associated with this repository.

> **Technical report:** *To be added.* The report will describe the data
> preparation procedure, distillation settings, training hyperparameters, and
> evaluation results.

## Training

The `train` directory contains the code required to train both model capacities:

1. `train_orthrus_fused_v40.py` — training of an Orthrus model based on
   Qwen3-0.6B.
2. `train_orthrus_fused_v42.py` — training of an Orthrus model based on
   Qwen3-1.7B.

Both scripts contain the training configuration and hyperparameters in the
`get_training_args()` function. For example:

```python
def get_training_args():
    training_args = {
        "base_model": "Qwen/Qwen3-1.7B",
        "dataset": "/path/to/DistillationDataset.Qwen3-1.7B.v29",
        "max_seq_len": 3072,
        "min_loss_tokens_per_block": 4,
        "block_size": 8,
        "num_blocks": 32,
        "batch_size": 10,
        "refinement_rate": 0.0,
        "confidence_weighting": "none",
        "confidence_temperature": 1.0,
        "ce_weight": 0.0,
        "training_objective": "ce",
        "epochs": 1,
        "dedicated_mask_token": "<|orthrus_mask|>",
        "save_diff": True,
        "eval_every": 8000,
        "num_eval_samples": 1000,
        "save_every": 2000,
        "attn": "eager",
        "out_dir": "/path/to/checkpoint-directory/orthrus-exp42"
    }
    return OrthrusTrainingConfig.from_json(training_args)
```

Before starting training, update the `dataset` and `out_dir` paths in the
corresponding script.

Training uses eight processes on eight GPUs. For example:

```bash
python -m torch.distributed.run --nproc_per_node=8 train_orthrus_fused_v42.py
```

Use `train_orthrus_fused_v40.py` instead when training the Qwen3-0.6B variant.

## Evaluation

The `eval` directory contains the code used to evaluate and benchmark trained
checkpoints:

- `eval_speed_orthrus.py` — measures **Tokens Per Forward (TPF)** and
  **Wallclock Tokens Per Second (TPS)**.
- `losslessness_evaluation.py` — evaluates Qwen--Orthrus generation trajectory
  matching.
- `evaluate-orthrus-via-lm_eval.py` — evaluates checkpoints using
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- `speed_evaluation_dataset.json` — dataset used for speed evaluation.

The analysis of Qwen--Orthrus trajectory matching, including the role of
numerical precision in the observed deviations, is presented in our preprint:

**[How Lossless Is Lossless Speculative Decoding? The Role of Numerical Precision in Orthrus](http://arxiv.org/abs/2609.15504)**

Training details, evaluation methodology, and the results reported in the
associated technical report will be added when the report is available.

## Reproducibility

To reproduce the training experiments:

1. Obtain the repository and the corresponding distilled dataset.
2. Install the required software dependencies and compatible CUDA/PyTorch
   versions.
3. Set the dataset and checkpoint output paths in the selected training script.
4. Start distributed training using the command shown above.
5. Use the scripts in `eval` to benchmark the resulting checkpoint.

Exact environment specifications, data preparation details, training
hyperparameters, and evaluation results will be documented in the technical
report.

## Citation

If you use this repository or the associated implementation, please cite the
Orthrus paper:

```bibtex
@article{orthrus2026,
  title   = {Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion},
  year    = {2026},
  eprint  = {2605.12825},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

For work concerning the losslessness analysis, please also refer to:

```bibtex
@article{losslessorthus2026,
  title   = {How Lossless Is Lossless Speculative Decoding? The Role of Numerical Precision in Orthrus},
  year    = {2026},
  eprint  = {2609.15504},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```
