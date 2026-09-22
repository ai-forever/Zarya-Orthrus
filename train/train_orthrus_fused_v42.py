import argparse
import contextlib
import copy
import os
import numpy as np
import pathlib
import random
import time
from typing import Any, List, Sequence, Dict, Tuple, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import tqdm
import datasets
from clearml import Task
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import transformers
from datetime import timedelta
import re

from fp32_logits import enable_fp32_logits, fp32_logits_metadata, FP32LogitsWrapper


def get_current_version():
    m = re.search(r"_v(\d+)\.py", __file__)
    if m:
        return m.group(1)
    else:
        return "X"


@dataclass
class OrthrusTrainingConfig:
    """
    Configuration for Orthrus training pipeline.
    """

    # Model & Architecture
    base_model: str = "Qwen/Qwen3-0.6B"
    # Which Orthrus backend to use for base_model: "auto" inspects the base
    # checkpoint's config.model_type at startup and picks the matching
    # backend module pair (model.py/configuration.py for Qwen3,
    # model_deepseekv3.py/configuration_deepseekv3.py for DeepSeek V3). Force
    # one explicitly only for debugging -- see resolve_orthrus_backend().
    backend: Literal["auto", "qwen3", "deepseek_v3"] = "auto"
    num_blocks: int = 10
    block_size: int = 32
    max_seq_len: int = 2048

    # Training
    epochs: int = 1
    batch_size: int = 1
    lr: float = 2e-4
    adam_eps: float = 1e-8
    seed: int = 31415926
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    param_dtype: Literal["same", "fp32"] = "same"

    # Objective & Loss
    objective: Literal["diffusion_kl"] = "diffusion_kl"
    ar_sample_temperature: float = 1e-4
    min_loss_tokens_per_step: int = 4
    min_loss_tokens_per_block: int = 4
    # Per-token loss weight w_k = exp(-(k-1)/gamma), k = 1-based offset from the
    # block anchor. Smaller gamma -> stronger down-weighting of trailing tokens
    # in a block. Default of 1000 makes w_k ~= 1 for any realistic block_size,
    # i.e. weighting is effectively a no-op unless explicitly set lower.
    gamma: float = 1000.0
    # If > 0, eval_generation_probe attempts a second diff-drafter pass ("refinement")
    # whenever a block's AR verification rejects only 1-3 tokens: unmask the AR-agreed
    # positions, re-mask only the disagreeing ones, and retry. This is purely an
    # eval-time inference-strategy toggle, unrelated to whether/how the refinement
    # training branch above is exercised.
    refinement_rate: float = 0.0

    # Batched Orthrus training: find the maximum spanning range of response
    # tokens shared by every sample in a local batch, place Orthrus blocks
    # once within it, and reuse that placement for every sample -- collapses
    # ar_seq_len to one scalar for the whole batch, letting a single forward
    # call replace what used to be batch_size sequential per-sample calls.
    # "auto" enables it whenever batch_size > 1 (a batch_size == 1 step
    # trivially degenerates to the old per-sample behavior either way);
    # force "on"/"off" only for debugging.
    enable_batched_diffusion: Literal["auto", "on", "off"] = "auto"

    # Confidence-aware KL weighting: down-weight the loss on tokens where the
    # teacher itself is not confident, so the student isn't forced to match
    # noisy/uncertain targets as hard as clear-cut ones. "none" disables this.
    #   "entropy":      w = exp(-H(teacher_softmax) / confidence_temperature)
    #   "prob_correct": w = teacher_prob(ground_truth_token) ** (1 / confidence_temperature)
    #   "margin":       w = sigmoid((logit(ground_truth) - logit(best competitor)) / confidence_temperature)
    confidence_weighting: Literal["none", "entropy", "prob_correct", "margin"] = "none"
    # Temperature controlling how sharply confidence maps to weight for whichever
    # confidence_weighting mode is selected; smaller -> more aggressive down-weighting
    # of low-confidence tokens. Unused when confidence_weighting == "none".
    confidence_temperature: float = 1.0
    # If > 0, adds ce_weight * CE(student_logits, ground_truth_tokens) on top of the
    # KL-vs-teacher-distribution loss: loss = kl_loss + ce_weight * ce_loss. This is a
    # direct hard-label cross-entropy term against the true next tokens, alongside the
    # soft-distillation KL term. 0.0 (default) disables it.
    ce_weight: float = 0.0
    # "kl": current approach -- frozen AR teacher forward + KL(student || teacher)
    # on masked positions (matches OrthrusAttention's shared-cache design; this is
    # what ce_weight/confidence_weighting above modulate).
    # "ce": plain cross-entropy of the student's diffusion-pass logits against the
    # true tokens directly, no teacher softmax computed at all. The teacher/AR
    # forward pass itself still runs (its transformer backbone populates the shared
    # KV cache the diffusion pass reads from -- see OrthrusAttention), but its
    # lm_head projection is skipped entirely (see teacher_logits_to_keep below), so
    # this saves the teacher-side lm_head matmul + softmax + kl_div, not the whole
    # teacher forward pass. confidence_weighting requires a teacher softmax and is
    # therefore rejected outright when training_objective="ce" (see main()).
    # gamma position-weighting has no such dependency (it's purely a function of
    # position-within-block) and stays available unchanged in both modes.
    training_objective: Literal["kl", "ce"] = "kl"

    # Data
    dataset: str = None  # required
    num_eval_samples: Optional[int] = None
    eval_loss_samples: Optional[int] = 200
    num_workers: int = 8

    # Paths & Output
    out_dir: Path = field(default_factory=lambda: Path("/home/jovyan/inkoziev/ckp/orthrus/orthrus_exp{}".format(get_current_version())))
    init_diff_state: Optional[Path] = None
    init_trainable_state: Optional[Path] = None

    # Checkpointing & Logging
    save_diff: bool = False
    save_every: int = 0
    log_every: int = 10
    eval_every: int = 4000

    # Mask Token
    mask_token_id: Optional[int] = None
    dedicated_mask_token: Optional[str] = None
    mask_init: Literal["mean", "eos", "pad"] = "mean"

    # Misc
    attn: str = "eager"
    trust_remote_code: bool = True
    skip_nonfinite: bool = True
    fp32_logits: bool = True
    promote_fp32_lm_head: bool = False

    # Config file reference (optional)
    config: Optional[str] = None

    def __post_init__(self):
        """Convert string paths to Path objects and do basic validation."""
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)
        if isinstance(self.init_diff_state, str):
            self.init_diff_state = Path(self.init_diff_state)
        if isinstance(self.init_trainable_state, str):
            self.init_trainable_state = Path(self.init_trainable_state)

    def save(self, path: Path):
        """Save config to JSON file."""
        import json
        data = {k: str(v) if isinstance(v, Path) else v
                for k, v in self.__dict__.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, data) -> "OrthrusTrainingConfig":
        # Convert string paths back to Path
        if "out_dir" in data:
            data["out_dir"] = Path(data["out_dir"])
        if "init_diff_state" in data and data["init_diff_state"]:
            data["init_diff_state"] = Path(data["init_diff_state"])
        if "init_trainable_state" in data and data["init_trainable_state"]:
            data["init_trainable_state"] = Path(data["init_trainable_state"])
        return cls(**data)


def get_training_args():
    training_args = {
        "base_model": "Qwen/Qwen3-1.7B",
        "dataset": "/home/jovyan/shares/SR008.fs2/data/DistillationDataset.Qwen3-1.7B.v29",
        "max_seq_len": 3072,
        "min_loss_tokens_per_block": 4,
        "block_size": 8,
        "num_blocks": 32,
        "batch_size": 10,
        "refinement_rate": 0.0,
        "confidence_weighting": "none",  # "entropy" | "prob_correct" | "margin" to try confidence-aware KL weighting
        "confidence_temperature": 1.0,
        "ce_weight": 0.0,  # e.g. 0.1 to add a hard-label CE term alongside the KL distillation loss
        "training_objective": "ce",  # "ce" for the KL-less, teacher-lm_head-free CE-only comparison run
        "epochs": 1,
        "dedicated_mask_token": "<|orthrus_mask|>",
        "save_diff": True,
        "eval_every": 8000,
        "num_eval_samples": 1000,
        "save_every": 2000,
        "attn": "eager",
        "out_dir": "/home/jovyan/shares/SR008.fs2/ckp/Orthrus/orthrus-exp42"
    }
    return OrthrusTrainingConfig.from_json(training_args)


# ==================== DISTRIBUTED UTILITIES ====================

def init_distributed_from_env() -> dict[str, int | bool]:
    """Initialize single-node/multi-node DDP when launched under torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=240))
        torch.cuda.set_device(local_rank)
    return {"distributed": distributed, "rank": rank, "local_rank": local_rank, "world_size": world_size}


def is_rank0(rank: int) -> bool:
    return rank == 0


def dist_mean_float(value: float, device: torch.device, distributed: bool, world_size: int) -> float:
    t = torch.tensor(float(value), dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
    return float(t.detach().cpu())


def dist_max_float(value: float, device: torch.device, distributed: bool) -> float:
    t = torch.tensor(float(value), dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.detach().cpu())


def dist_any_bool(value: bool, device: torch.device, distributed: bool) -> bool:
    t = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(int(t.detach().cpu()))


# ==================== DATASET & COLLATOR ====================

class OrthrusTrainDataset(Dataset):
    """
    Wrapper around Hugging Face datasets.Dataset for distributed training.
    Ensures deterministic shuffling per epoch via seed.
    """

    def __init__(self, hf_dataset, tokenizer, max_seq_len: int, block_size: int):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.block_size = block_size

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.hf_dataset[idx]
        prompt = record.get("prompt")
        response = record.get("response")

        messages = [{"role": "user", "content": prompt}]
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True
        )["input_ids"]

        response_tokens = self.tokenizer.encode(response + "<|im_end|>")

        input_ids = prompt_tokens + response_tokens

        # Truncate if too long (we skip in collator if still invalid)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        # Loss mask: 0 for prompt, 1 for response
        token_loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
        token_loss_mask = token_loss_mask[:self.max_seq_len]

        # # Pad to max_seq_len (?)
        # pad_len = self.max_seq_len - len(input_ids)
        # if pad_len > 0:
        #     pad_ids = [self.tokenizer.eos_token_id] * pad_len
        #     input_ids.extend(pad_ids)
        #     token_loss_mask.extend([0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "loss_mask": torch.tensor(token_loss_mask, dtype=torch.long),
            "seq_len": len(input_ids),  # actual length before padding (but we pad to fixed)
            "prompt_len": len(prompt_tokens),
            "response_len": len(response_tokens),
        }


def compute_sample_length_proxy(hf_dataset) -> List[int]:
    """
    Cheap length proxy for length-bucketed batching: character count of
    prompt+response. Deliberately avoids tokenizing the whole dataset just to
    sort it -- an approximation of token count, not exact, but computed via a
    single pass over raw strings rather than running the tokenizer (and chat
    template) over every example up front. Swap for real token counts later
    if the approximation ever turns out to matter.
    """
    prompts = hf_dataset["prompt"]
    responses = hf_dataset["response"]
    return [len(p or "") + len(r or "") for p, r in zip(prompts, responses)]


class LengthBucketedBatchSampler(Sampler):
    """
    Length-bucketed batch composition for the batched Orthrus diffusion path
    (see compute_batch_span/choose_batched_anchors below): grouping
    similarly-long samples into the same batch keeps their shared response
    span wide, which is what the batched path needs to actually place useful
    Orthrus blocks instead of falling back to the sequential path.

    Deterministic given (lengths, seed, epoch) -- every rank computes the
    *entire* global batch assignment independently and identically, then
    keeps only its own slice. No rank-0-only step, no broadcast: this
    mirrors how the base dataset order is already produced by
    `.shuffle(seed=args.seed)` identically on every rank.

    Batch *count* parity across ranks (required for DDP's lockstep
    collectives) is guaranteed structurally: samples are grouped into
    megabatches of size `batch_size * num_replicas`, shuffled at the
    megabatch level (this is what changes batch composition/order between
    epochs), and each megabatch is always split into exactly `num_replicas`
    equal `batch_size`-sized chunks -- one per rank. The final undersized
    megabatch (if the dataset doesn't divide evenly) is padded by cycling
    already-seen indices from within itself, so every rank always yields the
    same number of batches every epoch, same as DistributedSampler's own
    drop_last=False padding.
    """

    def __init__(self, lengths: List[int], batch_size: int, num_replicas: int, rank: int, seed: int):
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _megabatches(self) -> List[List[int]]:
        n = len(self.lengths)
        order = sorted(range(n), key=lambda i: self.lengths[i])
        mb_size = self.batch_size * self.num_replicas
        megabatches = [order[i:i + mb_size] for i in range(0, n, mb_size)]
        if megabatches and len(megabatches[-1]) < mb_size:
            last = list(megabatches[-1])
            i = 0
            while len(last) < mb_size:
                last.append(last[i % len(last)])
                i += 1
            megabatches[-1] = last
        g = random.Random(self.seed + self.epoch)
        g.shuffle(megabatches)
        return megabatches

    def __iter__(self):
        start = self.rank * self.batch_size
        for mb in self._megabatches():
            yield mb[start:start + self.batch_size]

    def __len__(self):
        mb_size = self.batch_size * self.num_replicas
        return (len(self.lengths) + mb_size - 1) // mb_size


class OrthrusCollator:
    """
    Collates a batch for Orthrus training: right-pads every sample's
    input_ids/loss_mask to the batch's max length (loss_mask is 0 on padding
    automatically, so padded positions never get trained on regardless of
    downstream anchor choice) and always returns a full batch of exactly
    len(batch) rows.

    IMPORTANT: this never returns None for a batch with no "valid" samples,
    unlike the previous version. A whole-batch None return meant `if batch
    is None: continue` in the training loop, which is only safe if every
    rank hits it on the exact same steps -- under DistributedSampler, each
    rank holds a disjoint shard, so there's no such guarantee, and ranks
    could silently desync on the collective calls inside masked_kl_loss.
    Individually short/invalid rows are instead handled downstream in the
    training loop by contributing zero weight rather than being dropped, so
    every rank always makes the same number of collective calls per step
    regardless of which specific rows are too short to use.
    """

    def __init__(self, block_size: int, min_response_len: int, pad_token_id: int):
        self.block_size = block_size
        self.min_response_len = min_response_len
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        max_len = max(item["seq_len"] for item in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        loss_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        prompt_lens, response_lens, seq_lens = [], [], []

        for i, item in enumerate(batch):
            L = item["seq_len"]
            input_ids[i, :L] = item["input_ids"]
            loss_mask[i, :L] = item["loss_mask"]
            prompt_lens.append(item["prompt_len"])
            response_lens.append(item["response_len"])
            seq_lens.append(L)

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "prompt_lens": prompt_lens,
            "response_lens": response_lens,
            "seq_lens": seq_lens,
            "batch_size": len(batch),
        }


# ==================== REMAINING HELPERS ====================

def dtype_from_arg(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def maybe_autocast(enabled: bool, dtype: torch.dtype):
    if enabled:
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.dtype):
        return str(x)
    return str(x)


def configure_mask_token(tokenizer, args: argparse.Namespace) -> dict[str, Any]:
    dedicated = getattr(args, "dedicated_mask_token", None)
    if dedicated:
        before = len(tokenizer)
        vocab = tokenizer.get_vocab()
        if dedicated not in vocab:
            tokenizer.add_special_tokens({"additional_special_tokens": [dedicated]})
        try:
            tokenizer.mask_token = dedicated
        except Exception:
            pass
        mask_token_id = tokenizer.convert_tokens_to_ids(dedicated)
        if mask_token_id is None or mask_token_id < 0:
            raise RuntimeError(f"Could not add/resolve dedicated mask token: {dedicated!r}")
        return {
            "mask_token_id": int(mask_token_id),
            "mask_token_text": dedicated,
            "dedicated_mask_token": dedicated,
            "added_tokens": len(tokenizer) - before,
            "tokenizer_len": len(tokenizer),
        }

    if args.mask_token_id is not None:
        mask_token_id = int(args.mask_token_id)
        return {
            "mask_token_id": mask_token_id,
            "mask_token_text": tokenizer.decode([mask_token_id]),
            "dedicated_mask_token": None,
            "added_tokens": 0,
            "tokenizer_len": len(tokenizer),
        }
    for candidate in [getattr(tokenizer, "mask_token_id", None), getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "eos_token_id", None)]:
        if candidate is not None:
            return {
                "mask_token_id": int(candidate),
                "mask_token_text": tokenizer.decode([int(candidate)]),
                "dedicated_mask_token": None,
                "added_tokens": 0,
                "tokenizer_len": len(tokenizer),
            }
    raise RuntimeError("Could not determine a mask token id; pass --mask-token-id or --dedicated-mask-token")


def resize_base_for_tokenizer(
    base: torch.nn.Module,
    tokenizer,
    mask_info: dict[str, Any],
    mask_init: str,
    extra_init_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    old_vocab = int(base.get_input_embeddings().weight.shape[0])
    mask_id = int(mask_info["mask_token_id"])
    new_vocab = max(int(len(tokenizer)), old_vocab, mask_id + 1)
    resized = new_vocab != old_vocab
    if resized:
        base.resize_token_embeddings(new_vocab)
    base.config.vocab_size = new_vocab

    init_source = "not_dedicated_mask"
    should_init_mask = bool(mask_info.get("dedicated_mask_token"))
    extra_init_token_ids = [int(i) for i in (extra_init_token_ids or [])]
    extra_init_sources: dict[int, str] = {}
    if should_init_mask or extra_init_token_ids:
        with torch.no_grad():
            in_weight = base.get_input_embeddings().weight
            out = base.get_output_embeddings()
            out_weight = out.weight if out is not None and out.weight is not in_weight else None
            if should_init_mask:
                init_source = _copy_row_from_source(in_weight, mask_id, mask_init, tokenizer)
                if out_weight is not None:
                    _copy_row_from_source(out_weight, mask_id, mask_init, tokenizer)
            for token_id in extra_init_token_ids:
                extra_init_sources[token_id] = _copy_row_from_source(in_weight, token_id, mask_init, tokenizer)
                if out_weight is not None:
                    _copy_row_from_source(out_weight, token_id, mask_init, tokenizer)
    return {
        "old_vocab_size": old_vocab,
        "tokenizer_len": int(len(tokenizer)),
        "new_vocab_size": new_vocab,
        "resized_token_embeddings": resized,
        "mask_embedding_init": init_source,
        "extra_token_embedding_init": extra_init_sources,
    }


def _copy_row_from_source(weight: torch.Tensor, row_id: int, source: str, tokenizer) -> str:
    if source == "mean":
        weight[row_id].copy_(weight[:row_id].float().mean(dim=0).to(dtype=weight.dtype))
        return "mean_existing_rows"
    source_id = getattr(tokenizer, f"{source}_token_id", None)
    if source_id is None:
        source_id = getattr(tokenizer, "eos_token_id", None)
    if source_id is None:
        weight[row_id].zero_()
        return "zeros_no_source_token"
    weight[row_id].copy_(weight[int(source_id)])
    return f"{source}_token_id:{int(source_id)}"


def ensure_qwen3_config_fields(cfg, block_size: int, mask_token_id: int):
    # Despite the name, this is backend-agnostic: it only sets fields every
    # supported backend's config declares (block_size, mask_token_id,
    # layer_types, _attn_implementation, _experts_implementation).
    cfg.block_size = block_size
    cfg.mask_token_id = mask_token_id
    if not hasattr(cfg, "layer_types") or cfg.layer_types is None:
        cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    if not hasattr(cfg, "_attn_implementation") or cfg._attn_implementation is None:
        cfg._attn_implementation = "eager"
    # MoE backends (e.g. DeepSeek V3) default _experts_implementation to
    # "grouped_mm". transformers validates that at PreTrainedModel.__init__ time
    # via a per-module source-code heuristic (_can_set_experts_implementation):
    # it checks whether the *defining module* of the model class mentions
    # "@use_experts_implementation" anywhere. Our Orthrus*LM classes live in
    # model_deepseekv3.py / model.py, which reuse the stock MoE module
    # unmodified rather than redefining it there, so that heuristic always
    # fails for us regardless of whether grouped_mm would actually work -- and
    # unlike attn_implementation, there's no silent fallback here when the
    # config's default is already the literal string "grouped_mm" (only a
    # None default degrades gracefully). Force "eager" explicitly so
    # PreTrainedModel.__init__ never even attempts the grouped_mm dispatch
    # check for our wrapper classes.
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"
    return cfg


def resolve_orthrus_backend(base_model: str, trust_remote_code: bool, override: str = "auto"):
    """
    Pick the Orthrus config/model classes for `base_model` by inspecting its
    config.json (config.model_type), and lazily import only the backend
    actually needed -- so e.g. running Qwen3 doesn't require
    model_deepseekv3.py to even be importable, and vice versa.

    Returns (OrthrusConfigClass, OrthrusLMClass, family) where family is
    "qwen3" or "deepseek_v3".
    """
    if override not in ("auto", "qwen3", "deepseek_v3"):
        raise ValueError(f"Unknown backend override={override!r}; expected 'auto', 'qwen3', or 'deepseek_v3'")

    if override == "auto":
        base_cfg = transformers.AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        model_type = getattr(base_cfg, "model_type", "")
        if model_type == "qwen3":
            family = "qwen3"
        elif model_type == "deepseek_v3":
            family = "deepseek_v3"
        else:
            raise ValueError(
                f"Don't know which Orthrus backend to use for base_model={base_model!r} "
                f"(config.model_type={model_type!r}); pass --backend qwen3|deepseek_v3 to force one."
            )
    else:
        family = override

    if family == "deepseek_v3":
        from configuration_deepseekv3 import OrthrusDeepseekV3Config
        from model_deepseekv3 import OrthrusDeepseekV3LM
        return OrthrusDeepseekV3Config, OrthrusDeepseekV3LM, family
    else:
        from configuration import OrthrusConfig
        from model import OrthrusLM
        return OrthrusConfig, OrthrusLM, family


def copy_ar_weights_from_base(orthrus: torch.nn.Module, base: torch.nn.Module):
    missing, unexpected = orthrus.load_state_dict(base.state_dict(), strict=False)
    return list(missing), list(unexpected)


def init_diff_from_ar(model: torch.nn.Module) -> int:
    copied = 0
    for _name, mod in model.named_modules():
        pairs = [
            # Qwen3 attention (model.py: OrthrusAttention)
            ("q_proj", "q_proj_diff"),
            ("k_proj", "k_proj_diff"),
            ("v_proj", "v_proj_diff"),
            ("o_proj", "o_proj_diff"),
            ("q_norm", "q_norm_diff"),
            ("k_norm", "k_norm_diff"),
            # DeepSeek V3 MLA (model_deepseekv3.py: OrthrusDeepseekV3Attention)
            # -- "q_proj" above already covers the q_lora_rank=None path;
            # these cover the low-rank q_lora_rank path plus the KV path.
            ("q_a_proj", "q_a_proj_diff"),
            ("q_a_layernorm", "q_a_layernorm_diff"),
            ("q_b_proj", "q_b_proj_diff"),
            ("kv_a_proj_with_mqa", "kv_a_proj_with_mqa_diff"),
            ("kv_a_layernorm", "kv_a_layernorm_diff"),
            ("kv_b_proj", "kv_b_proj_diff"),
        ]
        for src_name, dst_name in pairs:
            if hasattr(mod, src_name) and hasattr(mod, dst_name):
                src = getattr(mod, src_name)
                dst = getattr(mod, dst_name)
                if hasattr(src, "weight") and hasattr(dst, "weight") and src.weight.shape == dst.weight.shape:
                    dst.weight.data.copy_(src.weight.data)
                    copied += 1
                if getattr(src, "bias", None) is not None and getattr(dst, "bias", None) is not None and src.bias.shape == dst.bias.shape:
                    dst.bias.data.copy_(src.bias.data)
                    copied += 1
    return copied


def trainable_diff_params(model: torch.nn.Module):
    params = []
    names = []
    for name, p in model.named_parameters():
        if "_diff" in name:
            p.requires_grad_(True)
            params.append(p)
            names.append(name)
        else:
            p.requires_grad_(False)
    return names, params


def add_trainable_vocab_token_rows(model: torch.nn.Module, token_ids: Sequence[int]) -> tuple[list[str], list[torch.nn.Parameter], dict[str, Any]]:
    ids = sorted({int(i) for i in token_ids})
    names: list[str] = []
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    candidates: list[tuple[str, torch.nn.Parameter | None]] = []
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embed is not None:
        candidates.append(("model.embed_tokens.weight", getattr(embed, "weight", None)))
    head = getattr(model, "lm_head", None)
    if head is not None:
        candidates.append(("lm_head.weight", getattr(head, "weight", None)))

    for name, param in candidates:
        if param is None or id(param) in seen:
            continue
        if max(ids, default=-1) >= int(param.shape[0]):
            raise RuntimeError(f"Cannot train vocab rows {ids}; {name} has only {param.shape[0]} rows")
        param.requires_grad_(True)
        param.register_hook(_vocab_row_gradient_hook(ids))
        names.append(name)
        params.append(param)
        seen.add(id(param))
    return names, params, {"enabled": bool(params), "token_ids": ids, "parameter_names": names}


def _vocab_row_gradient_hook(token_ids: Sequence[int]):
    ids = sorted({int(i) for i in token_ids})

    def hook(grad: torch.Tensor) -> torch.Tensor:
        if grad is None or grad.ndim < 2:
            return grad
        mask = torch.zeros((grad.shape[0],), dtype=grad.dtype, device=grad.device)
        valid = [i for i in ids if 0 <= i < grad.shape[0]]
        if valid:
            mask[torch.tensor(valid, dtype=torch.long, device=grad.device)] = 1
        return grad * mask.view(-1, *([1] * (grad.ndim - 1)))

    return hook


def compute_batch_span(
    prompt_lens: List[int],
    response_lens: List[int],
    block_size: int,
    min_span_tokens: int,
) -> Optional[Tuple[int, int]]:
    """
    lo = latest point at which every sample's prompt has ended (max over the
    batch); hi = earliest point at which any sample's response ends (min
    over the batch). [lo, hi) is real response content for EVERY sample in
    the batch simultaneously -- guaranteed by construction, since lo is
    >= every prompt_len and hi is <= every prompt_len + response_len.

    Returns None (degenerate) if that shared span can't fit at least one
    block plus the configured minimum -- the caller should fall back to the
    per-sample sequential path for this step. Note this single check also
    subsumes "some sample's response is too short to use": a near-empty
    response directly shrinks hi, which shows up here as a small/negative
    span rather than needing a separate per-sample validity check.
    """
    lo = max(prompt_lens)
    hi = min(p + r for p, r in zip(prompt_lens, response_lens))
    if hi - lo < max(block_size, min_span_tokens):
        return None
    return lo, hi


def choose_batched_anchors(lo: int, hi: int, block_size: int, num_blocks: int, rng: random.Random) -> List[int]:
    """
    Anchors shared across every sample in the batch -- valid because [lo, hi)
    is real response content for all of them (see compute_batch_span).
    num_blocks is adapted down to however many non-overlapping-enough anchor
    positions actually fit in the available span, rather than assumed fixed;
    a narrow shared span means fewer anchors this step, not a crash or an
    out-of-range sample.
    """
    max_anchor = hi - block_size
    available = max_anchor - lo + 1
    adaptive_num_blocks = max(1, min(num_blocks, available))
    return sorted(rng.sample(range(lo, max_anchor + 1), adaptive_num_blocks))


def build_diffusion_batch_batched(input_ids: torch.Tensor, anchors: List[int], block_size: int, mask_token_id: int):
    """
    Batched analogue of build_diffusion_batch: input_ids is (B, seq_len) and
    the SAME anchors are used for every row. Only valid when every row's
    tokens at each anchor position are real (non-padding) content, which
    compute_batch_span guarantees by construction.
    """
    B = input_ids.shape[0]
    device = input_ids.device
    blocks, pos, causal = [], [], []
    for a in anchors:
        block = torch.full((B, block_size), mask_token_id, dtype=torch.long, device=device)
        block[:, 0] = input_ids[:, a]
        blocks.append(block)
        pos.append(torch.arange(a, a + block_size, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1))
        causal.append(torch.full((B, block_size), a - 1, dtype=torch.long, device=device))
    return torch.cat(blocks, dim=1), torch.cat(pos, dim=1), torch.cat(causal, dim=1)


def packed_anchors(seq_len: int, block_size: int, num_blocks: int, rng: random.Random) -> List[int]:
    max_anchor = seq_len - block_size
    if max_anchor < 1:
        raise ValueError(f"seq_len={seq_len} too small for block_size={block_size}")
    if num_blocks <= max_anchor:
        return sorted(rng.sample(range(1, max_anchor + 1), num_blocks))
    return sorted(rng.randrange(1, max_anchor + 1) for _ in range(num_blocks))


def build_diffusion_batch(input_ids: torch.Tensor, anchors: List[int], block_size: int, mask_token_id: int):
    device = input_ids.device
    blocks = []
    pos = []
    causal = []
    for a in anchors:
        block = torch.full((block_size,), mask_token_id, dtype=torch.long, device=device)
        block[0] = input_ids[0, a]
        blocks.append(block)
        pos.append(torch.arange(a, a + block_size, dtype=torch.long, device=device))
        causal.append(torch.full((block_size,), a - 1, dtype=torch.long, device=device))
    return (
        torch.cat(blocks, dim=0).unsqueeze(0),
        torch.cat(pos, dim=0).unsqueeze(0),
        torch.cat(causal, dim=0).unsqueeze(0),
    )


def build_refinement_diffusion_batch(input_with_masking: torch.Tensor, anchors: List[int], block_size: int):
    # ND: unlike `build_diffusion_batch`, the block content is NOT re-masked from scratch here.
    # `input_with_masking` must already carry the correct ground-truth tokens at the "accepted"
    # positions and mask_token_id at the positions that still need re-drafting (see
    # choose_refinement_anchor) -- we just slice it out. The other difference from
    # `build_diffusion_batch` is the causal limit: here it's the block end (not the anchor),
    # so the diff drafter can attend across the whole partially-unmasked block.
    device = input_with_masking.device
    blocks = []
    pos = []
    causal = []
    for a in anchors:
        block = input_with_masking[0, a:a + block_size].clone()
        blocks.append(block)
        pos.append(torch.arange(a, a + block_size, dtype=torch.long, device=device))
        causal.append(torch.full((block_size,), a + block_size - 1, dtype=torch.long, device=device))
    return (
        torch.cat(blocks, dim=0).unsqueeze(0),
        torch.cat(pos, dim=0).unsqueeze(0),
        torch.cat(causal, dim=0).unsqueeze(0),
    )


def gather_teacher_logits(teacher_logits: torch.Tensor, anchors: List[int], block_size: int) -> torch.Tensor:
    idx = []
    for a in anchors:
        idx.extend(range(a, a + block_size - 1))
    return teacher_logits[:, idx, :]


def gather_target_token_ids(input_ids: torch.Tensor, anchors: List[int], block_size: int) -> torch.Tensor:
    """
    Ground-truth token ids at the same a+1 .. a+block_size-1 positions gathered
    by gather_target_loss_mask / gather_student_logits / gather_teacher_logits.
    Always index into the TRUE (unmasked) input sequence, never a masked copy --
    these are the labels the student's diffusion output is ultimately judged
    against (for confidence weighting and/or the CE loss term).
    """
    idx = []
    for a in anchors:
        idx.extend(range(a + 1, a + block_size))
    return input_ids[:, idx]


def gather_student_logits(diff_logits: torch.Tensor, num_blocks: int, block_size: int) -> torch.Tensor:
    rows = []
    for b in range(num_blocks):
        base = b * block_size
        rows.extend(range(base, base + block_size - 1))
    return diff_logits[:, rows, :]


def block_loss_weights(block_size: int, gamma: float, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Per-block, per-position loss weight w_k = exp(-(k-1)/gamma), where k is the
    1-based token offset from the block anchor (k = 1 .. block_size - 1).
    Heading tokens in a block (small k) get weight close to 1; trailing tokens
    are down-weighted as k grows. Shape: (block_size - 1,), aligned with the
    a+1 .. a+block_size-1 label-shift convention used throughout (index i
    corresponds to k = i + 1).
    """
    k = torch.arange(1, block_size, dtype=dtype, device=device)
    return torch.exp(-(k - 1) / gamma)


def gather_target_loss_mask(
    loss_mask: torch.Tensor,
    anchors: List[int],
    block_size: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    idx = []
    for a in anchors:
        idx.extend(range(a + 1, a + block_size))
    mask = loss_mask[:, idx].float()
    if weights is not None:
        # weights has shape (block_size - 1,); repeat once per anchor block so
        # the k-offset resets at the start of every block, then broadcast over
        # the batch dim.
        tiled = weights.repeat(len(anchors)).unsqueeze(0)
        mask = mask * tiled
    return mask


def choose_masked_anchors(
    seq_len: int,
    block_size: int,
    num_blocks: int,
    rng: random.Random,
    loss_mask: torch.Tensor | None,
    min_loss_tokens_per_block: int,
) -> List[int]:
    if loss_mask is None or min_loss_tokens_per_block <= 0:
        return packed_anchors(seq_len, block_size, num_blocks, rng)
    host_mask = loss_mask[0].detach().cpu()
    max_anchor = seq_len - block_size
    eligible = [
        a for a in range(1, max_anchor + 1)
        if int(host_mask[a].item()) > 0
        and int(host_mask[a + 1 : a + block_size].sum().item()) >= min_loss_tokens_per_block
    ]
    if not eligible:
        return packed_anchors(seq_len, block_size, num_blocks, rng)
    if num_blocks <= len(eligible):
        return sorted(rng.sample(eligible, num_blocks))
    return sorted(rng.choice(eligible) for _ in range(num_blocks))


def choose_refinement_anchor(
        seq_len: int,
        block_size: int,
        rng: random.Random,
        sample_input: torch.Tensor,
        loss_mask: torch.Tensor,
        min_loss_tokens_per_block: int,
        mask_token_id: int,
        weights: torch.Tensor | None = None) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Choose a single anchor and simulate a partially-verified draft block: a
    random subset of the block's non-anchor positions (not necessarily a
    contiguous run) is treated as already AR-agreed and left as ground
    truth; the rest are replaced with MASK, as the tokens the diff drafter
    still needs to (re-)predict. See the eval_generation_probe refinement
    pass for why this need not be a contiguous suffix: there, "agreement"
    is decided per-position (diff draft token == AR's own greedy guess at
    that position), so the resulting mismatch pattern can be scattered,
    not just a trailing run from the first mismatch.

    Returns:
        anchors: one-element list [a]
        input_with_masking: sample_input with the re-masked positions
            replaced by mask_token_id (same shape as sample_input)
        target_loss_mask: (1, block_size - 1) float tensor, aligned with
            the a+1 .. a+block_size-1 label-shift convention used by
            gather_target_loss_mask / gather_teacher_logits /
            gather_student_logits -- 1 at re-masked positions that also
            fall in the response region, 0 elsewhere (including the
            already-accepted positions, since they're already known and
            shouldn't be trained on again).
    """
    device = sample_input.device
    host_mask = loss_mask[0].detach().cpu()
    max_anchor = seq_len - block_size
    eligible = [
        a for a in range(1, max_anchor + 1)
        if int(host_mask[a].item()) > 0
        and int(host_mask[a + 1: a + block_size].sum().item()) >= min_loss_tokens_per_block
    ]
    if not eligible:
        # Don't hard-fail a batch just because no anchor satisfies
        # min_loss_tokens_per_block; fall back to any valid anchor position.
        eligible = list(range(1, max_anchor + 1))

    # Choose the single anchor.
    a = rng.choice(eligible)
    anchors = [a]

    # Number of "mismatching" tokens to re-mask; at least 1 accepted non-anchor
    # token must remain in the block, so k ranges over 1 .. block_size - 2.
    num_block_masks = rng.randint(1, block_size - 2)

    # Random positions except the anchor (use the seeded rng, not the global
    # `random` module, so this stays reproducible and doesn't desync ranks).
    mask_idx = rng.sample(range(a + 1, a + block_size), num_block_masks)
    mask_idx = torch.tensor(mask_idx, dtype=torch.long, device=device)

    # Clone the input sequence because we modify it by replacing the mismatching
    # positions with MASK; the accepted positions (and everything outside the
    # block) keep their ground-truth token.
    input_with_masking = sample_input.detach().clone()
    input_with_masking[:, mask_idx] = mask_token_id

    # Loss mask: 1 only at re-masked positions (the ones the drafter must actually
    # predict), 0 at the already-accepted prefix -- using the same label-shift
    # convention as gather_target_loss_mask (index i <-> absolute position a+1+i).
    target_loss_mask = torch.zeros((1, block_size - 1), dtype=torch.float32, device=device)
    rel_idx = mask_idx - (a + 1)
    target_loss_mask[0, rel_idx] = 1.0

    # Also restrict to the response region and apply the gamma position weighting,
    # same as the ordinary branch does.
    response_mask = gather_target_loss_mask(loss_mask, anchors, block_size, weights=weights)
    target_loss_mask = target_loss_mask * response_mask

    return anchors, input_with_masking, target_loss_mask


def teacher_confidence_weights(
    teacher_logits: torch.Tensor,
    target_ids: torch.Tensor | None,
    mode: str,
    temperature: float,
) -> torch.Tensor:
    """
    Per-token confidence weight derived from the teacher's distribution at each
    masked position, higher confidence -> larger weight. `teacher_logits` and
    `target_ids` are treated as constants here (detached) -- this is a fixed
    per-token reweighting of the student's loss, not something the student's
    gradient should flow back through.

    mode:
      "entropy":      w = exp(-H / temperature), H = Shannon entropy of the
                      teacher's softmax over the full vocab. Low entropy
                      (peaked/confident distribution) -> w close to 1.
      "prob_correct": w = p(ground_truth) ** (1 / temperature). temperature=1
                      makes this just the teacher's probability on the true
                      next token.
      "margin":       margin = logit(ground_truth) - logit(best competing
                      token). w = sigmoid(margin / temperature) -- mapped
                      through a sigmoid (not a raw exp) so the weight stays
                      bounded in (0, 1) regardless of how large the logit
                      margin gets.
    """
    logits = teacher_logits.float().detach()
    log_probs = F.log_softmax(logits, dim=-1)

    if mode == "entropy":
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return torch.exp(-entropy / temperature)

    if target_ids is None:
        raise ValueError(f"confidence_weighting={mode!r} requires target_ids (ground-truth tokens)")
    target_ids = target_ids.detach()

    if mode == "prob_correct":
        gt_log_prob = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return torch.exp(gt_log_prob / temperature)

    if mode == "margin":
        gt_logit = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        logits_wo_gt = logits.scatter(-1, target_ids.unsqueeze(-1), float("-inf"))
        competitor_logit = logits_wo_gt.max(dim=-1).values
        margin = gt_logit - competitor_logit
        return torch.sigmoid(margin / temperature)

    raise ValueError(f"unknown confidence_weighting mode: {mode!r}")


def teacher_logits_to_keep(training_objective: str, device: torch.device):
    """
    logits_to_keep value for the teacher/AR forward call. In "ce" mode the
    teacher's lm_head projection isn't needed at all -- only its transformer
    backbone, whose output populates the shared KV cache the diffusion pass
    reads from (see OrthrusAttention) -- so a zero-length index tensor skips
    the lm_head matmul over the teacher's whole sequence entirely. In "kl"
    mode the default (0 -> "keep every position") is used, since
    gather_teacher_logits needs real logits at every position.
    """
    if training_objective == "ce":
        return torch.zeros(0, dtype=torch.long, device=device)
    return 0


def masked_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    target_mask: torch.Tensor | None,
    device: torch.device,
    distributed: bool,
    world_size: int,
    target_ids: torch.Tensor | None = None,
    confidence_mode: str = "none",
    confidence_temperature: float = 1.0,
    ce_weight: float = 0.0,
    training_objective: str = "kl",
) -> tuple[torch.Tensor, float, float]:
    if training_objective not in ("kl", "ce"):
        raise ValueError(f"unknown training_objective: {training_objective!r}")
    if training_objective == "ce" and confidence_mode != "none":
        # Enforced here too (not just at startup in main()) since this function
        # is also called directly by eval_loss_probe.
        raise ValueError(
            "confidence_weighting requires a teacher softmax, which training_objective="
            "'ce' does not compute; set confidence_weighting='none' for ce runs."
        )

    if training_objective == "ce":
        if target_ids is None:
            raise ValueError("training_objective='ce' requires target_ids (ground-truth tokens)")
        vocab = student_logits.shape[-1]
        per_pos_loss = F.cross_entropy(
            student_logits.float().reshape(-1, vocab), target_ids.reshape(-1), reduction="none"
        ).reshape(target_ids.shape)
    else:
        if teacher_logits is None:
            raise ValueError("training_objective='kl' requires teacher_logits")
        per_pos_loss = F.kl_div(
            F.log_softmax(student_logits.float(), dim=-1),
            F.softmax(teacher_logits.float(), dim=-1),
            reduction="none",
        ).sum(dim=-1)

    if target_mask is None:
        # Unweighted global-mean fallback path (kept for parity with the pre-mask API).
        loss = per_pos_loss.mean()
        if training_objective == "kl" and ce_weight > 0:
            if target_ids is None:
                raise ValueError("ce_weight > 0 requires target_ids (ground-truth tokens)")
            vocab = student_logits.shape[-1]
            ce = F.cross_entropy(student_logits.float().reshape(-1, vocab), target_ids.reshape(-1))
            loss = loss + ce_weight * ce
        return loss, float(student_logits.shape[1]), float(student_logits.shape[1])

    mask = target_mask.to(device=per_pos_loss.device, dtype=per_pos_loss.dtype)

    # Confidence-aware re-weighting: fold a per-token teacher-confidence weight
    # into the same mask used for gamma/response-region gating, so it feeds
    # through the existing weighted-average machinery below unchanged. Only
    # meaningful (and only allowed, per the check above) in "kl" mode.
    if training_objective == "kl" and confidence_mode != "none":
        conf_w = teacher_confidence_weights(teacher_logits, target_ids, confidence_mode, confidence_temperature)
        mask = mask * conf_w.to(device=mask.device, dtype=mask.dtype)

    local_num = (per_pos_loss * mask).sum()
    local_den = mask.sum()

    if training_objective == "kl" and ce_weight > 0:
        vocab = student_logits.shape[-1]
        ce_per_pos = F.cross_entropy(
            student_logits.float().reshape(-1, vocab),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(target_ids.shape)
        local_ce_num = (ce_per_pos * mask).sum()
    else:
        local_ce_num = torch.zeros_like(local_num)

    # Single combined all-reduce for [num, ce_num, den] instead of one per term.
    stats = torch.stack([local_num, local_ce_num, local_den]).detach().clone().to(device=device, dtype=torch.float32)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    global_num_t, global_ce_num_t, global_den_t = stats[0], stats[1], stats[2]
    global_den = global_den_t.clamp_min(1.0)

    if distributed:
        main_loss = local_num * float(world_size) / global_den
        ce_loss = local_ce_num * float(world_size) / global_den
    else:
        main_loss = local_num / global_den
        ce_loss = local_ce_num / global_den

    # ce_weight only ever combines with the "kl" objective (an additional hard-label
    # term alongside distillation); in "ce" mode main_loss already *is* the CE loss.
    loss = main_loss + ce_weight * ce_loss if (training_objective == "kl" and ce_weight > 0) else main_loss
    return loss, float(global_den_t.detach().cpu()), float(global_num_t.detach().cpu())


def safe_clip_grad_norm_(params: list[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        device = params[0].device if params else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    device = grads[0].device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for g in grads:
        gf = g.detach().float()
        total = total + torch.sum(gf * gf)
    total_norm = torch.sqrt(total)
    if torch.isfinite(total_norm) and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(scale.to(dtype=g.dtype))
    return total_norm


def save_hf_checkpoint(model: torch.nn.Module, tokenizer, path: Path) -> None:
    """
    Save a portable, standard HF checkpoint.

    enable_fp32_logits() renames the real lm_head Linear to
    `lm_head.wrapped`, which breaks OrthrusLM's class-level
    `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`
    lookup used by save_pretrained. We temporarily restore the plain
    Linear at `model.lm_head` for the save (so "lm_head.weight" is a
    real path again), then put the wrapper back so training continues
    unaffected.

    If promote_head was used, the lm_head weight is a genuinely separate
    fp32 tensor (no longer tied to the embeddings), so we also permanently
    tell the model/config it's untied -- otherwise save_pretrained would
    treat it as a duplicate of the embedding and silently drop it.
    """
    head = model.lm_head
    if isinstance(head, FP32LogitsWrapper):
        if head.promoted and model.config.tie_word_embeddings:
            # Tie was broken for real when the head was promoted to fp32.
            # This is a permanent fact about the model from here on, not
            # just a save-time toggle.
            model.config.tie_word_embeddings = False
            model._tied_weights_keys = {}
            print("Note: promote_fp32_lm_head broke weight tying; "
                  "marking lm_head as untied going forward.")

        model.lm_head = head.wrapped   # restore "lm_head.weight" path
        try:
            path.mkdir(exist_ok=True, parents=True)
            tokenizer.save_pretrained(path)
            model.save_pretrained(path, ignore_metadata_errors=True)
        finally:
            model.lm_head = head       # put the fp32 wrapper back
    else:
        path.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(path)
        model.save_pretrained(path, ignore_metadata_errors=True)


def consecutive_true_count(flags: torch.Tensor) -> int:
    if flags.numel() == 0:
        return 0
    return int(flags.long().cumprod(dim=0).sum().item())


@torch.inference_mode()
def eval_generation_probe(model, tokenizer, dataset, args, rng):
    """Calculate realistic metrics of decoding including the acceptance length for speculative decoding."""
    model.eval()
    device = model.device
    records = []
    outputs = []

    for pidx, record in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
        prompt = record.get("prompt")
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(messages,
                                                   add_generation_prompt=True,
                                                   enable_thinking=False).input_ids
        max_new_tokens = 96

        prompt_tokens = len(prompt_ids)
        input_ids = torch.LongTensor(prompt_ids).unsqueeze(0).to(device)
        output_ids = torch.full(
            (1, prompt_tokens + max_new_tokens + args.block_size + 2),
            model.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        output_ids[:, :prompt_tokens] = input_ids
        past_key_values = DynamicCache(config=model.config)
        pos = torch.arange(prompt_tokens, device=device).unsqueeze(0)
        t0 = time.time()
        ar = model(input_ids=input_ids, position_ids=pos, past_key_values=past_key_values, use_cache=True, is_diffusion_pass=False)
        start_idx = prompt_tokens
        next_token = ar.logits[:, -1, :].argmax(dim=-1)
        output_ids[:, start_idx] = next_token
        target_len = prompt_tokens + max_new_tokens
        block_records = []
        while start_idx < target_len - 1:
            diff_len = min(args.block_size, target_len - start_idx)
            diff_block_ids = torch.full((1, diff_len), model.config.mask_token_id, dtype=torch.long, device=device)
            diff_block_ids[:, 0] = output_ids[:, start_idx]
            diff_pos = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)
            diff = model(
                input_ids=diff_block_ids,
                position_ids=diff_pos,
                past_key_values=past_key_values,
                use_cache=False,
                is_diffusion_pass=True,
                ar_seq_len=start_idx,
            )
            if diff_len > 1:
                diff_tokens = diff.logits[:, :-1, :].argmax(dim=-1)
            else:
                diff_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
            proposed_block = torch.cat([output_ids[:, start_idx : start_idx + 1], diff_tokens], dim=1)
            ar_verify = model(
                input_ids=proposed_block,
                position_ids=diff_pos,
                past_key_values=past_key_values,
                use_cache=True,
                is_diffusion_pass=False,
            )
            ar_tokens = ar_verify.logits.argmax(dim=-1)
            matches = diff_tokens.eq(ar_tokens[:, :-1]) if diff_tokens.numel() else torch.empty((1, 0), dtype=torch.bool, device=device)

            refined = False
            if args.refinement_rate > 0 and diff_tokens.numel():
                num_mismatches = int(diff_tokens.numel()) - int(matches[0].sum().item())
                if 1 <= num_mismatches <= 3:
                    # 1) Roll the kv-cache back to just before this block: drop the
                    #    diff_len entries that the ar_verify call above just appended,
                    #    since we're about to redraft and re-verify the whole block.
                    past_key_values.crop(start_idx)

                    # 2) Copy the original (all-MASK-except-anchor) block template and
                    #    reveal the AR-agreed positions with their AR-verified token;
                    #    positions that disagreed stay MASK.
                    refined_block_ids = diff_block_ids.clone()
                    refined_block_ids[:, 1:][matches] = ar_tokens[:, :-1][matches]

                    # 3) Re-run the diff drafter over the partially unmasked block,
                    #    letting it attend across the whole block (revealed tokens
                    #    and remaining masks alike).
                    refined_diff = model(
                        input_ids=refined_block_ids,
                        position_ids=diff_pos,
                        past_key_values=past_key_values,
                        use_cache=False,
                        is_diffusion_pass=True,
                        ar_seq_len=start_idx + diff_len,
                    )
                    refined_diff_tokens = refined_diff.logits[:, :-1, :].argmax(dim=-1)
                    refined_proposed_block = torch.cat(
                        [output_ids[:, start_idx:start_idx + 1], refined_diff_tokens], dim=1
                    )

                    # 4) Re-verify with AR exactly as in the original algorithm.
                    refined_ar_verify = model(
                        input_ids=refined_proposed_block,
                        position_ids=diff_pos,
                        past_key_values=past_key_values,
                        use_cache=True,
                        is_diffusion_pass=False,
                    )
                    refined_ar_tokens = refined_ar_verify.logits.argmax(dim=-1)
                    refined_matches = refined_diff_tokens.eq(refined_ar_tokens[:, :-1])

                    # Adopt the refined draft/verification for the accept/commit logic below.
                    diff_tokens = refined_diff_tokens
                    proposed_block = refined_proposed_block
                    ar_verify = refined_ar_verify
                    ar_tokens = refined_ar_tokens
                    matches = refined_matches
                    refined = True

            teacher_match_rate = matches[0].sum().item() / (matches[0].shape[0]+1e-5)
            accepted = consecutive_true_count(matches[0])
            correction = ar_tokens[:, accepted]
            end_idx = start_idx + accepted + 1
            accepted_block = proposed_block[:, :accepted + 1]
            output_ids[:, start_idx:end_idx] = accepted_block
            past_key_values.crop(end_idx)
            block_records.append({
                "start_idx": int(start_idx),
                "diff_len": int(diff_len),
                "proposed_tokens": int(diff_len - 1),
                "matches": matches[0].tolist(),
                "teacher_match_rate": teacher_match_rate,
                "accepted_proposals": int(accepted),
                "committed_tokens_including_anchor": int(accepted + 1),
                "full_accept": bool(accepted == diff_len - 1),
                "refined": bool(refined),
            })
            start_idx = end_idx
            if start_idx < target_len:
                output_ids[:, start_idx] = correction
        elapsed = time.time() - t0
        generated = output_ids[:, :target_len]
        outputs.append({
            "prompt_index": pidx,
            "elapsed_s": elapsed,
            "new_tokens": int(target_len - prompt_tokens),
            "text_sample": tokenizer.decode(generated[0, :min(target_len, prompt_tokens + 80)].tolist(), skip_special_tokens=False),
        })
        records.extend(block_records)

    model.train()
    accepted = [r["accepted_proposals"] for r in records]
    committed = [r["committed_tokens_including_anchor"] for r in records]
    hist = {str(i): accepted.count(i) for i in range(args.block_size)}

    # mean_matches[i] indicates the probability of token in position i proposed by diff drafter to be accepted by AR verifier.
    matching_matrix = np.asarray([r["matches"] for r in records if len(r["matches"])==args.block_size-1])
    mean_matches = np.mean(matching_matrix, axis=0)

    denorm = max(1, len(records))
    # number of blocks with exactly 1, 2, 3 mispredicted (i.e. AR-rejected) tokens
    misprediction1_rate = sum([len(r["matches"]) - sum(r["matches"]) == 1 for r in records]) / denorm
    misprediction2_rate = sum([len(r["matches"]) - sum(r["matches"]) == 2 for r in records]) / denorm
    misprediction3_rate = sum([len(r["matches"]) - sum(r["matches"]) == 3 for r in records]) / denorm
    applied_refinement_rate = sum(1 for r in records if r["refined"]) / denorm

    return {
        "max_proposable_tokens": args.block_size - 1,
        "num_diffusion_iterations": len(records),
        "mean_teacher_match_rate": np.mean([r["teacher_match_rate"] for r in records]),
        "mean_accepted_proposals": sum(accepted) / max(1, len(accepted)),
        "max_accepted_proposals": max(accepted) if accepted else 0,
        "mean_committed_tokens_including_anchor": sum(committed) / max(1, len(committed)),
        "mean_matches": mean_matches.tolist(),
        "full_accept_rate": sum(1 for r in records if r["full_accept"]) / max(1, len(records)),
        "misprediction1_rate": misprediction1_rate,
        "misprediction2_rate": misprediction2_rate,
        "misprediction3_rate": misprediction3_rate,
        "applied_refinement_rate": applied_refinement_rate,
        "hist_accepted_proposals": hist,
        "first_blocks": records,
        "outputs": outputs,
    }


@torch.inference_mode()
def eval_loss_probe(
    model,
    tokenizer,
    dataset,
    args,
    mask_token_id: int,
    rng: random.Random,
    autocast_enabled: bool,
    dtype: torch.dtype,
    max_samples: Optional[int] = None,
    block_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    """
    Compute the same masked-KL diffusion loss used in training, on held-out
    eval data. Mirrors the per-sample computation in the training loop
    (teacher AR forward -> diffusion student forward -> masked_kl_loss),
    just without gradients and without the DDP wrapper. Applies the same
    gamma position weighting as training so eval_loss reflects the actual
    objective being optimized.
    """
    model.eval()
    device = model.device

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    weighted_loss_sum = 0.0
    total_tokens = 0
    num_samples = 0

    for idx in tqdm.tqdm(range(n), desc="Eval loss", total=n):
        record = dataset[idx]
        prompt = record.get("prompt")
        response = record.get("response")
        if not prompt or not response:
            continue

        messages = [{"role": "user", "content": prompt}]
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, return_dict=True
        )["input_ids"]
        response_tokens = tokenizer.encode(response + "<|im_end|>")

        # Same min-response-length filter the training collator applies.
        if len(response_tokens) < args.block_size:
            continue

        input_ids_list = (prompt_tokens + response_tokens)[: args.max_seq_len]
        loss_mask_list = ([0] * len(prompt_tokens) + [1] * len(response_tokens))[: args.max_seq_len]
        seq_len = len(input_ids_list)
        if seq_len - args.block_size < 1:
            continue  # too short for even one anchor block

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device).unsqueeze(0)
        sample_mask = torch.tensor(loss_mask_list, dtype=torch.long, device=device).unsqueeze(0)

        best_anchors = choose_masked_anchors(
            seq_len, args.block_size, args.num_blocks, rng, sample_mask, args.min_loss_tokens_per_block
        )
        target_loss_mask = gather_target_loss_mask(sample_mask, best_anchors, args.block_size, weights=block_weights)
        target_ids = gather_target_token_ids(input_ids, best_anchors, args.block_size)

        ar_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        with maybe_autocast(autocast_enabled, dtype):
            teacher = model(
                input_ids=input_ids, position_ids=ar_pos, use_cache=True, is_diffusion_pass=False,
                logits_to_keep=teacher_logits_to_keep(args.training_objective, device),
            )
            teacher_logits = None
            if args.training_objective == "kl":
                teacher_logits = gather_teacher_logits(teacher.logits, best_anchors, args.block_size)
                if autocast_enabled:
                    teacher_logits = teacher_logits.to(dtype)
            past_key_values = teacher.past_key_values

        diff_ids, diff_pos, causal_limit = build_diffusion_batch(input_ids, best_anchors, args.block_size, mask_token_id)

        with maybe_autocast(autocast_enabled, dtype):
            diff_out = model(
                input_ids=diff_ids,
                position_ids=diff_pos,
                past_key_values=past_key_values,
                use_cache=False,
                is_diffusion_pass=True,
                causal_limit=causal_limit,
                ar_seq_len=seq_len,
            )
            student_logits = gather_student_logits(diff_out.logits, args.num_blocks, args.block_size)
            if autocast_enabled:
                student_logits = student_logits.to(dtype)

        # distributed=False: this runs on rank 0 only, so no cross-rank reduction needed.
        loss, tokens, _ = masked_kl_loss(
            student_logits, teacher_logits, target_loss_mask, device, distributed=False, world_size=1,
            target_ids=target_ids,
            confidence_mode=args.confidence_weighting,
            confidence_temperature=args.confidence_temperature,
            training_objective=args.training_objective,
            ce_weight=args.ce_weight,
        )
        loss_val = float(loss.detach().cpu())
        if not (loss_val == loss_val):  # NaN guard
            continue

        weighted_loss_sum += loss_val * tokens
        total_tokens += tokens
        num_samples += 1

    model.train()

    if total_tokens <= 0:
        return {"eval_loss": float("nan"), "eval_loss_tokens": 0, "eval_loss_samples": 0}

    return {
        "eval_loss": weighted_loss_sum / total_tokens,
        "eval_loss_tokens": int(total_tokens),
        "eval_loss_samples": num_samples,
    }


def init_clearml(args) -> Optional["Task"]:
    if Task is None:
        return None
    task = Task.init(
        project_name=os.getenv("CLEARML_PROJECT", "Orthrus"),
        task_name=os.getenv("CLEARML_TASK_NAME", pathlib.Path(__file__).stem),
        tags=[t.strip() for t in os.getenv("CLEARML_TAGS", "").split(",") if t.strip()],
    )
    task.connect(vars(args))
    return task


def main() -> int:
    args = get_training_args()

    if args.training_objective == "ce" and args.confidence_weighting != "none":
        raise ValueError(
            f"confidence_weighting={args.confidence_weighting!r} requires a teacher softmax, "
            f"which training_objective='ce' does not compute; set confidence_weighting='none' "
            f"(or use training_objective='kl') for confidence-aware weighting."
        )

    ddp_info = init_distributed_from_env()
    rank = ddp_info["rank"]
    local_rank = ddp_info["local_rank"]
    world_size = ddp_info["world_size"]
    distributed = ddp_info["distributed"]

    # rank must be known before this: previously called unconditionally (and before
    # ddp_info even existed), so every worker created its own ClearML Task -- see
    # the bug report this fixed.
    clearml_task = init_clearml(args) if is_rank0(rank) else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + rank * 1_000_003)
    torch.manual_seed(args.seed + rank * 1_000_003)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = dtype_from_arg(args.dtype)
    param_dtype = torch.float32 if args.param_dtype == "fp32" else dtype
    autocast_enabled = (param_dtype == torch.float32 and dtype in (torch.float16, torch.bfloat16))

    if is_rank0(rank):
        print(json.dumps({
            "event": "startup",
            "args": vars(args),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "distributed": distributed,
            "cuda_device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
        }, ensure_ascii=False, default=json_default), flush=True)

    # Tokenizer & Mask
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    mask_info = configure_mask_token(tokenizer, args)
    mask_token_id = int(mask_info["mask_token_id"])
    if is_rank0(rank):
        print(f"mask_token_id={mask_token_id}")

    # Load HF datasets
    train_hf = datasets.load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    eval_hf = datasets.load_dataset(args.dataset, split="test").shuffle(seed=args.seed)
    if args.num_eval_samples is not None and len(eval_hf) > args.num_eval_samples:
        eval_hf = eval_hf.take(args.num_eval_samples)

    # Wrap datasets
    train_dataset = OrthrusTrainDataset(train_hf, tokenizer, args.max_seq_len, args.block_size)
    #eval_dataset = OrthrusTrainDataset(eval_hf, tokenizer, args.max_seq_len, args.block_size)  # reuse wrapper for consistency

    collator = OrthrusCollator(
        block_size=args.block_size, min_response_len=args.block_size,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
    )

    batched_diffusion_enabled = {"auto": args.batch_size > 1, "on": True, "off": False}[args.enable_batched_diffusion]

    if batched_diffusion_enabled:
        # Length-bucketed batch composition (see LengthBucketedBatchSampler):
        # computed independently and deterministically on every rank, no
        # broadcast. Requires knowing sample lengths up front, so this uses
        # the raw HF dataset directly rather than the tokenizing Dataset
        # wrapper -- a cheap proxy (see compute_sample_length_proxy), not a
        # full tokenization pass.
        length_proxy = compute_sample_length_proxy(train_hf)
        batch_sampler = LengthBucketedBatchSampler(
            length_proxy, batch_size=args.batch_size, num_replicas=world_size, rank=rank, seed=args.seed,
        )
        sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        # Unchanged from before: plain per-rank sharding, batch_size == 1
        # (or forced off), same as prior to the batching work.
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        ) if distributed else None
        batch_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    OrthrusConfigCls, OrthrusLMCls, backend_family = resolve_orthrus_backend(
        args.base_model, args.trust_remote_code, override=args.backend
    )
    if is_rank0(rank):
        print(f"Orthrus backend: {backend_family} (base_model={args.base_model})")

    # Base model loading (only on rank 0 to save memory, then broadcast weights)
    if is_rank0(rank):
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
            #torch_dtype=param_dtype,
            dtype=param_dtype,
            attn_implementation=args.attn,
        )
        cfg = ensure_qwen3_config_fields(copy.deepcopy(base.config), args.block_size, mask_token_id)
        resize_info = resize_base_for_tokenizer(
            base, tokenizer, mask_info, args.mask_init, extra_init_token_ids=[]
        )
        model = OrthrusLMCls(cfg)
        missing, unexpected = copy_ar_weights_from_base(model, base)
        diff_copies = init_diff_from_ar(model)
        del base
    else:
        cfg = None
        resize_info = None
        missing = unexpected = None
        diff_copies = None

    # Broadcast config and create model on all ranks
    if distributed:
        if is_rank0(rank):
            cfg_dict = {
                k: v for k, v in vars(cfg).items()
                if (not k.startswith('_') or k in ("_attn_implementation", "_experts_implementation"))
            }
            object_list = [cfg_dict]
        else:
            object_list = [None]

        dist.broadcast_object_list(object_list, src=0, device=device)
        cfg_dict = object_list[0]

        if not is_rank0(rank):
            cfg = OrthrusConfigCls(**cfg_dict)

        dist.barrier()

    if not is_rank0(rank):
        model = OrthrusLMCls(cfg)
        missing, unexpected = [], []
        diff_copies = init_diff_from_ar(model)  # will be overwritten by DDP sync

    model.to(dtype=param_dtype, device=device)

    logits_info = enable_fp32_logits(model, promote_head=args.promote_fp32_lm_head) if args.fp32_logits else fp32_logits_metadata(model)

    trainable_names, trainable_params = trainable_diff_params(model)
    if not trainable_params:
        raise RuntimeError("No trainable *_diff parameters found")

    # Load init state if provided (only rank 0 loads, then DDP syncs)
    init_trainable_state_info = None
    if args.init_trainable_state is not None or args.init_diff_state is not None:
        init_path = args.init_trainable_state or args.init_diff_state
        if is_rank0(rank):
            state = torch.load(init_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"Loaded init trainable state from {init_path}")
        if distributed:
            for p in model.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=0)
        init_trainable_state_info = {"path": str(init_path), "loaded": True}

    # DDP wrapper
    train_model: torch.nn.Module = model
    if distributed:
        train_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0, eps=args.adam_eps)

    # Per-token position weight w_k = exp(-(k-1)/gamma) applied to the loss of every
    # draft block (k = 1-based offset from the block's anchor). Computed once since
    # it depends only on block_size/gamma, not on any per-step data.
    block_weights = block_loss_weights(args.block_size, args.gamma, device)

    # Metadata
    metadata = {
        "event": "model_ready",
        "base_model": args.base_model,
        "mask_token_id": mask_token_id,
        "mask_token_text": mask_info["mask_token_text"],
        "mask_info": mask_info,
        "resize_info": resize_info if is_rank0(rank) else None,
        "max_seq_len": args.max_seq_len,
        "num_blocks": args.num_blocks,
        "block_size": args.block_size,
        "objective": args.objective,
        "dataset": args.dataset,
        "trainable_param_count": sum(p.numel() for p in trainable_params),
        "trainable_tensor_count": len(trainable_params),
        "first_trainable_names": trainable_names[:12],
        "distributed": distributed,
        "world_size": world_size,
        "effective_batch_size_per_gpu": args.batch_size,
    }

    if is_rank0(rank):
        print(json.dumps(metadata, ensure_ascii=False, default=json_default), flush=True)
        with open(args.out_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False, default=json_default)

        args.save(args.out_dir / "training_args.json")

        if clearml_task is not None:
            with open(__file__) as f:
                code_text = f.read()
            clearml_task.upload_artifact(pathlib.Path(__file__).name, artifact_object=code_text)
            clearml_task.upload_artifact("metadata", artifact_object=metadata, auto_pickle=False, extension_name=".json")

    step = 0
    losses: List[float] = []

    evaluation_metrics_fp = args.out_dir / "evaluation_metrics.jsonl"

    # Deterministic, rank-agreed refinement scheduling (see the training-loop
    # body): whether a given step uses the refinement branch is a function of
    # the global step counter alone, identical on every rank without any
    # synchronization -- unlike a per-rank rng.random() draw, which could
    # disagree across ranks and desync the collective calls inside
    # masked_kl_loss (sequential mode does one call per sample; batched mode
    # does one call total -- a mismatch in which mode different ranks pick
    # for the same step is exactly the failure mode this avoids).
    refinement_period = round(1.0 / args.refinement_rate) if args.refinement_rate > 0 else None

    for epoch in range(args.epochs):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        elif distributed:
            sampler.set_epoch(epoch)  # ensure proper shuffling every epoch

        for batch_idx, batch in enumerate(train_loader):
            t0 = time.time()
            train_model.train()

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            token_loss_mask = batch["loss_mask"].to(device, non_blocking=True)

            padded_len = input_ids.shape[1]

            # Refinement scheduling is decided ONCE per step (not per sample,
            # and not via a per-rank rng draw -- see refinement_period's
            # definition above for why). When it fires, EVERY sample this
            # step uses the refinement branch, and the batched fast path is
            # skipped entirely for this step (refinement isn't implemented
            # there -- see the batching-plan discussion): this preserves the
            # *volume* of refinement training relative to the old per-sample
            # random draw, just redistributed into occasional all-refinement
            # steps instead of spread thin every step.
            use_refinement = (refinement_period is not None) and (step % refinement_period == 0)
            attempt_batched = batched_diffusion_enabled and not use_refinement

            batch_plan = None
            if attempt_batched:
                batch_plan = compute_batch_span(
                    batch["prompt_lens"], batch["response_lens"],
                    args.block_size, args.min_loss_tokens_per_block + 1,
                )

            # The batched-window-too-narrow decision is data-dependent (a function
            # of THIS rank's local batch content), so it must be agreed across
            # ranks before anyone commits to a code path -- otherwise one rank
            # could do a single batched masked_kl_loss call while another does
            # batch_size sequential calls for the same step, desyncing the
            # collectives inside masked_kl_loss. attempt_batched itself needs no
            # such sync since it's a pure function of (args, step), identical on
            # every rank already.
            run_batched = False
            if attempt_batched:
                any_degenerate = dist_any_bool(batch_plan is None, device, distributed)
                run_batched = not any_degenerate

            total_loss = 0.0
            num_valid_samples = 0

            if run_batched:
                lo, hi = batch_plan
                best_anchors = choose_batched_anchors(lo, hi, args.block_size, args.num_blocks, rng)
                target_loss_mask = gather_target_loss_mask(token_loss_mask, best_anchors, args.block_size, weights=block_weights)

                ar_pos = torch.arange(padded_len, device=device).unsqueeze(0).expand(input_ids.shape[0], -1)
                with torch.no_grad(), maybe_autocast(autocast_enabled, dtype):
                    teacher = model(
                        input_ids=input_ids,
                        position_ids=ar_pos,
                        use_cache=True,
                        is_diffusion_pass=False,
                        logits_to_keep=teacher_logits_to_keep(args.training_objective, device),
                    )
                    teacher_logits = None
                    if args.training_objective == "kl":
                        teacher_logits = gather_teacher_logits(teacher.logits.detach(), best_anchors, args.block_size)
                        if autocast_enabled:
                            teacher_logits = teacher_logits.to(dtype)
                    past_key_values = teacher.past_key_values

                diff_ids, diff_pos, causal_limit = build_diffusion_batch_batched(input_ids, best_anchors, args.block_size, mask_token_id)

                with maybe_autocast(autocast_enabled, dtype):
                    diff_out = train_model(
                        input_ids=diff_ids,
                        position_ids=diff_pos,
                        past_key_values=past_key_values,
                        use_cache=False,
                        is_diffusion_pass=True,
                        causal_limit=causal_limit,
                        ar_seq_len=padded_len,
                    )

                student_logits = gather_student_logits(diff_out.logits, len(best_anchors), args.block_size)
                if autocast_enabled:
                    student_logits = student_logits.to(dtype)

                target_ids = gather_target_token_ids(input_ids, best_anchors, args.block_size)
                loss, loss_tokens_global, _ = masked_kl_loss(
                    student_logits, teacher_logits, target_loss_mask,
                    device, distributed, world_size,
                    target_ids=target_ids,
                    confidence_mode=args.confidence_weighting,
                    confidence_temperature=args.confidence_temperature,
                    ce_weight=args.ce_weight,
                    training_objective=args.training_objective,
                )
                total_loss += loss
                num_valid_samples += 1
            else:
                # Sequential per-sample path: batch_size == 1, this step drew
                # refinement, or the batched window degenerated on some rank
                # this step (all ranks fall back together, per run_batched
                # above -- never just the rank that hit it).
                for b_idx in range(input_ids.shape[0]):
                    sample_input = input_ids[b_idx:b_idx+1]
                    sample_mask = token_loss_mask[b_idx:b_idx+1]

                    if use_refinement:
                        # Rare branch training the diff drafter to work on a partially
                        # unmasked block context. For simplicity, only one block with
                        # partially masked input is created.
                        best_anchors, input_with_masking, target_loss_mask = choose_refinement_anchor(
                            padded_len, args.block_size, rng,
                            sample_input, sample_mask, args.min_loss_tokens_per_block,
                            mask_token_id, weights=block_weights
                        )

                        # Forward passes. The teacher/AR pass must see the TRUE, unmasked sequence --
                        # it provides the ground-truth distillation target -- never the partially
                        # masked one (that's only the *input* to the diffusion/student pass below).
                        ar_pos = torch.arange(padded_len, device=device).unsqueeze(0)
                        with torch.no_grad(), maybe_autocast(autocast_enabled, dtype):
                            teacher = model(
                                input_ids=sample_input,
                                position_ids=ar_pos,
                                use_cache=True,
                                is_diffusion_pass=False,
                                logits_to_keep=teacher_logits_to_keep(args.training_objective, device),
                            )
                            teacher_logits = None
                            if args.training_objective == "kl":
                                teacher_logits = gather_teacher_logits(teacher.logits.detach(), best_anchors, args.block_size)
                                if autocast_enabled:
                                    teacher_logits = teacher_logits.to(dtype)
                            past_key_values = teacher.past_key_values

                        # causal_limit is different for this pass: diff drafter must attend to the partially (un)masked block!
                        diff_ids, diff_pos, causal_limit = build_refinement_diffusion_batch(input_with_masking, best_anchors, args.block_size)

                        with maybe_autocast(autocast_enabled, dtype):
                            diff_out = train_model(
                                input_ids=diff_ids,
                                position_ids=diff_pos,
                                past_key_values=past_key_values,
                                use_cache=False,
                                is_diffusion_pass=True,
                                causal_limit=causal_limit,
                                ar_seq_len=padded_len,
                            )

                        # The rest is the same as for ordinary pass with many blocks.
                        student_logits = gather_student_logits(diff_out.logits, args.num_blocks, args.block_size)
                        if autocast_enabled:
                            student_logits = student_logits.to(dtype)

                        target_ids = gather_target_token_ids(sample_input, best_anchors, args.block_size)
                        loss, loss_tokens_global, _ = masked_kl_loss(
                            student_logits, teacher_logits, target_loss_mask,
                            device, distributed, world_size,
                            target_ids=target_ids,
                            confidence_mode=args.confidence_weighting,
                            confidence_temperature=args.confidence_temperature,
                            ce_weight=args.ce_weight,
                            training_objective=args.training_objective,
                        )
                        total_loss += loss
                        num_valid_samples += 1
                    else:
                        # Ordinary Orthrus loss calculation pass with several completely masked blocks.
                        best_anchors = choose_masked_anchors(
                            padded_len, args.block_size, args.num_blocks, rng,
                            sample_mask, args.min_loss_tokens_per_block
                        )

                        target_loss_mask = gather_target_loss_mask(sample_mask, best_anchors, args.block_size, weights=block_weights)

                        # Forward passes
                        ar_pos = torch.arange(padded_len, device=device).unsqueeze(0)
                        with torch.no_grad(), maybe_autocast(autocast_enabled, dtype):
                            teacher = model(
                                input_ids=sample_input,
                                position_ids=ar_pos,
                                use_cache=True,
                                is_diffusion_pass=False,
                                logits_to_keep=teacher_logits_to_keep(args.training_objective, device),
                            )
                            teacher_logits = None
                            if args.training_objective == "kl":
                                teacher_logits = gather_teacher_logits(teacher.logits.detach(), best_anchors, args.block_size)
                                if autocast_enabled:
                                    teacher_logits = teacher_logits.to(dtype)
                            past_key_values = teacher.past_key_values

                        diff_ids, diff_pos, causal_limit = build_diffusion_batch(sample_input, best_anchors, args.block_size, mask_token_id)

                        with maybe_autocast(autocast_enabled, dtype):
                            diff_out = train_model(
                                input_ids=diff_ids,
                                position_ids=diff_pos,
                                past_key_values=past_key_values,
                                use_cache=False,
                                is_diffusion_pass=True,
                                causal_limit=causal_limit,
                                ar_seq_len=padded_len,
                            )

                        student_logits = gather_student_logits(diff_out.logits, args.num_blocks, args.block_size)
                        if autocast_enabled:
                            student_logits = student_logits.to(dtype)

                        target_ids = gather_target_token_ids(sample_input, best_anchors, args.block_size)
                        loss, loss_tokens_global, _ = masked_kl_loss(
                            student_logits, teacher_logits, target_loss_mask,
                            device, distributed, world_size,
                            target_ids=target_ids,
                            confidence_mode=args.confidence_weighting,
                            confidence_temperature=args.confidence_temperature,
                            ce_weight=args.ce_weight,
                            training_objective=args.training_objective,
                        )
                        total_loss += loss
                        num_valid_samples += 1

            loss = total_loss / num_valid_samples

            opt.zero_grad(set_to_none=True)
            skipped_step = False
            loss_nonfinite = dist_any_bool(not bool(torch.isfinite(loss).detach().cpu()), device, distributed)

            if args.skip_nonfinite and loss_nonfinite:
                grad_norm = torch.tensor(float("nan"), device=device)
                skipped_step = True
            else:
                loss.backward()
                grad_norm = safe_clip_grad_norm_(trainable_params, max_norm=1.0)
                grad_nonfinite = dist_any_bool(not bool(torch.isfinite(grad_norm).detach().cpu()), device, distributed)
                if args.skip_nonfinite and grad_nonfinite:
                    opt.zero_grad(set_to_none=True)
                    skipped_step = True
                else:
                    opt.step()

            loss_mean = dist_mean_float(float(loss.detach().cpu()), device, distributed, world_size)
            grad_norm_max = dist_max_float(float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm), device, distributed)
            peak_mem_max = dist_max_float(torch.cuda.max_memory_allocated() / 1024**3, device, distributed)

            step += 1

            if is_rank0(rank):
                losses.append(loss_mean)

            if is_rank0(rank) and (step == 1 or step % args.log_every == 0):
                print(json.dumps({
                    "event": "step",
                    "step": step,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss": loss_mean,
                    "grad_norm_max": grad_norm_max,
                    "skipped_step": skipped_step,
                    "elapsed_s": round(time.time() - t0, 3),
                    "cuda_mem_gb_max": round(peak_mem_max, 3),
                    "world_size": world_size,
                }, default=json_default), flush=True)

                if clearml_task and clearml_task.get_logger():
                    logger = clearml_task.get_logger()
                    logger.report_scalar("loss", "train", value=loss_mean, iteration=step)
                    logger.report_scalar("grad_norm_max", "train", value=grad_norm_max, iteration=step)

            # Evaluation (rank 0 only)
            if is_rank0(rank) and (step % args.eval_every == 0):
                loss_results = eval_loss_probe(
                    model, tokenizer, eval_hf, args, mask_token_id,
                    random.Random(args.seed), autocast_enabled, dtype,
                    max_samples=args.eval_loss_samples,
                    block_weights=block_weights,
                )
                eval_results = eval_generation_probe(model, tokenizer, eval_hf, args, random.Random(args.seed))
                eval_metrics = {
                    "eval_loss": loss_results["eval_loss"],
                    "eval_loss_tokens": loss_results["eval_loss_tokens"],
                    "eval_loss_samples": loss_results["eval_loss_samples"],
                    "mean_teacher_match_rate": eval_results["mean_teacher_match_rate"],
                    "mean_accepted_proposals": eval_results["mean_accepted_proposals"],
                    "max_accepted_proposals": eval_results["max_accepted_proposals"],
                    "mean_committed_tokens_including_anchor": eval_results["mean_committed_tokens_including_anchor"],
                    "full_accept_rate": eval_results["full_accept_rate"],
                    "misprediction1_rate": eval_results["misprediction1_rate"],
                    "misprediction2_rate": eval_results["misprediction2_rate"],
                    "misprediction3_rate": eval_results["misprediction3_rate"],
                    "applied_refinement_rate": eval_results["applied_refinement_rate"]
                }
                print("[evaluation results]", eval_metrics)

                with open(evaluation_metrics_fp, "a+") as f:
                    f.write(json.dumps(eval_metrics) + "\n")

                if clearml_task and clearml_task.get_logger():
                    for metric, score in eval_metrics.items():
                        clearml_task.get_logger().report_scalar(metric, "eval", value=score, iteration=step)

                eval_dir = args.out_dir / f"evaluation_{step}"
                eval_dir.mkdir(exist_ok=True)
                with open(eval_dir/"evaluation_results.json", "w") as f:
                    json.dump(eval_results, f, indent=4, ensure_ascii=False)

                with open(eval_dir/"eval_metrics.json", "w") as f:
                    json.dump(eval_metrics, f, indent=4, ensure_ascii=False)

            if is_rank0(rank) and args.save_every > 0 and step % args.save_every == 0:
                ckpt_dir = args.out_dir / f"checkpoint_{step}"
                print(f"Saving checkpoint to {ckpt_dir}...")
                ckpt_dir.mkdir(exist_ok=True)

                # only the diff layers weights
                ckpt_path = ckpt_dir / f"diff_state.pt"
                train_state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
                torch.save(train_state, ckpt_path)

                # transformers-compatible checkpoint
                hf_ckpt_path = ckpt_dir / "hf_ckp"
                save_hf_checkpoint(model, tokenizer, hf_ckpt_path)

                print(f"Checkpoint saved: {ckpt_path}")

    # Final save
    if is_rank0(rank):
        final_state_path = args.out_dir / "diff_state.pt"
        train_state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
        torch.save(train_state, final_state_path)

        ckp_dir = args.out_dir / "hf_ckp"
        save_hf_checkpoint(model, tokenizer, ckp_dir)
        print(f"Final model saved to {ckp_dir}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
