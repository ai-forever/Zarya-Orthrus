"""Utilities for forcing Orthrus/Qwen lm_head logits to be computed in fp32.

Low-precision checkpoints often run the final lm_head matmul in bf16/fp16 and
return quantized logits. That is fine for throughput, but it pollutes equivalence
checks and distillation targets with output-quantization artifacts. This helper
wraps only the output projection: hidden states and lm_head parameters are cast
to fp32 for the final linear projection, and the returned logits stay fp32.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class FP32LogitsWrapper(torch.nn.Module):
    """Wrap a language-model output head and compute logits in fp32.

    The wrapped module's parameters remain in their original dtype unless
    ``enable_fp32_logits(..., promote_head=True)`` is requested. Keeping the
    parameters in low precision preserves memory use while still avoiding bf16 or
    fp16 accumulation/output quantization in the final projection.
    """

    def __init__(self, wrapped: torch.nn.Module, *, promoted: bool = False):
        super().__init__()
        if isinstance(wrapped, FP32LogitsWrapper):
            # Idempotence: reuse the existing underlying module and remember if
            # either call asked for promotion.
            self.wrapped = wrapped.wrapped
            self.promoted = bool(wrapped.promoted or promoted)
        else:
            self.wrapped = wrapped
            self.promoted = bool(promoted)
        self.original_head_class = f"{self.wrapped.__class__.__module__}.{self.wrapped.__class__.__name__}"

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.wrapped.weight  # type: ignore[attr-defined]

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return getattr(self.wrapped, "bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.linear(hidden_states.float(), weight, bias)

    def extra_repr(self) -> str:
        return f"wrapped={self.original_head_class}, promoted={self.promoted}"


def _get_lm_head(model: torch.nn.Module) -> torch.nn.Module:
    head = getattr(model, "lm_head", None)
    if head is None and hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    if head is None:
        raise AttributeError(f"{model.__class__.__name__} has no lm_head/get_output_embeddings()")
    return head


def enable_fp32_logits(model: torch.nn.Module, *, promote_head: bool = False) -> dict[str, Any]:
    """Wrap ``model.lm_head`` so final logits are computed and returned as fp32.

    Args:
        model: Causal LM/Orthrus model with an ``lm_head`` output projection.
        promote_head: If true, convert the lm_head parameters to fp32 once before
            wrapping. If false, parameters are left in their current dtype and are
            cast to fp32 inside each forward.

    Returns:
        JSON-serializable metadata describing the effective configuration.
    """
    head = _get_lm_head(model)
    if isinstance(head, FP32LogitsWrapper):
        if promote_head and not head.promoted:
            head.wrapped.to(dtype=torch.float32)
            head.promoted = True
        wrapped = head
    else:
        if promote_head:
            head.to(dtype=torch.float32)
        wrapped = FP32LogitsWrapper(head, promoted=promote_head)
        if hasattr(model, "set_output_embeddings"):
            try:
                model.set_output_embeddings(wrapped)
            except Exception:
                setattr(model, "lm_head", wrapped)
        else:
            setattr(model, "lm_head", wrapped)
        # Some model implementations do not route set_output_embeddings() to the
        # public lm_head attribute. Keep both views consistent when possible.
        if hasattr(model, "lm_head") and getattr(model, "lm_head") is not wrapped:
            setattr(model, "lm_head", wrapped)

    return fp32_logits_metadata(model)


def fp32_logits_metadata(model: torch.nn.Module) -> dict[str, Any]:
    """Return JSON-serializable status for the fp32 logits wrapper."""
    try:
        head = _get_lm_head(model)
    except AttributeError:
        return {"fp32_logits": False, "reason": "no_lm_head"}
    enabled = isinstance(head, FP32LogitsWrapper)
    inner = head.wrapped if enabled else head  # type: ignore[union-attr]
    weight = getattr(inner, "weight", None)
    bias = getattr(inner, "bias", None)
    return {
        "fp32_logits": enabled,
        "promote_head": bool(getattr(head, "promoted", False)) if enabled else False,
        "head_class": f"{head.__class__.__module__}.{head.__class__.__name__}",
        "wrapped_head_class": getattr(head, "original_head_class", None) if enabled else None,
        "weight_dtype": str(weight.dtype) if weight is not None else None,
        "bias_dtype": str(bias.dtype) if bias is not None else None,
        "output_dtype": "torch.float32" if enabled else str(weight.dtype) if weight is not None else None,
    }
