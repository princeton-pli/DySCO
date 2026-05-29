"""FlashAttention-compatible attention paths for DySCO's decode loop.

DySCO's decode loop has two attention shapes that the FA2 public API does not
support out of the box:

  (a) The real pass applies a key-side additive bias to the pre-softmax
      logits. `flash_attn_func` accepts no `attn_bias` argument.
  (b) The speculative pass and context-warmup pass need the attention
      weight vectors of pre-detected QRHeads. `flash_attn_func` does not
      return per-head attention weights.

Forcing these onto eager attention is functionally correct but expensive in
both compute and memory. This module provides two drop-in replacements that
keep the bulk of the attention computation in FA.

------------------------------------------------------------------------------
Path 1 -- biased attention via augmented head dimension
    build_augmented_qk(), fa_intervention_attention_forward()

We encode the key-side bias into a single extra head-dim channel:

    Q'[q] = [Q[q] ; 1]
    K'[k] = [K[k] ; bias[k] * sqrt(d)]
    V'[k] = [V[k] ; 0]

so that (Q'.K') / sqrt(d) = (Q.K)/sqrt(d) + bias[k] exactly. With
`softmax_scale` overridden to 1/sqrt(d), the unmodified FA2 kernel computes
the biased attention, and the augmented output channel is identically zero
(V padded with 0) so slicing [..., :d] recovers the desired result. The
trick is mathematically exact, not an approximation.

------------------------------------------------------------------------------
Path 2 -- vanilla FA plus side-channel QRHead snapshot
    materialize_qrhead_attn(), fa_spec_attention_forward()

Only a small pre-selected set of (layer, head) attention vectors is needed
downstream of the speculative / warmup forwards, not the full all-head
attention matrix. So we:

  1. run vanilla FA to produce the main attention output -- this still uses
     all heads, since that output feeds the next layer's input;
  2. for ONLY the QRHeads of the current layer, additionally compute
     softmax(Q[h] . K[kv(h)]^T / sqrt(d)) and write the resulting (B, L_k)
     vector into a shared dict keyed by (layer_idx, head_idx).

The caller (dysco/custom_mixin.py::_rescale_sample) builds a fresh empty dict
before each speculative or warmup forward, passes it down as a kwarg, and
reads the populated dict back after the forward returns.

------------------------------------------------------------------------------
Conventions used throughout this module:

  * Inputs `query` (B, H_q, L_q, d) and `key`/`value` (B, H_kv, L_k, d) are
    in torch (B, H, L, D) layout. `flash_attn_func` expects (B, L, H, D), so
    we transpose at the call boundary.
  * FA handles GQA natively when H_kv < H_q -- no `repeat_kv` is performed.
  * Both paths run softmax (or its equivalent) in fp32 and cast back to the
    query dtype, matching the eager implementation's precision behavior.
"""

import math

import torch


def build_augmented_qk(query, key, value, key_bias, head_dim):
    """Augment Q/K/V to head_dim d+1 so vanilla FA computes biased attention.

    Args:
        query: (B, H_q,  L_q, d)  torch layout (B, H, L, D)
        key:   (B, H_kv, L_k, d)
        value: (B, H_kv, L_k, d)
        key_bias: (B, L_k) additive log-bias, broadcast across heads and queries
        head_dim: d (original head dimension, before augmentation)

    Returns:
        (Q_aug, K_aug, V_aug), each with head_dim d+1.
    """
    B, H_q, L_q, _ = query.shape
    _, H_kv, L_k, _ = key.shape

    q_extra = torch.ones((B, H_q, L_q, 1), device=query.device, dtype=query.dtype)
    # bias[k] * sqrt(d) goes into K's extra channel; broadcast (B, L_k) -> (B, H_kv, L_k, 1)
    k_extra = (key_bias.to(key.dtype) * math.sqrt(head_dim))[:, None, :, None].expand(B, H_kv, L_k, 1)
    v_extra = torch.zeros((B, H_kv, L_k, 1), device=value.device, dtype=value.dtype)

    q_aug = torch.cat([query, q_extra], dim=-1)
    k_aug = torch.cat([key, k_extra], dim=-1)
    v_aug = torch.cat([value, v_extra], dim=-1)
    return q_aug, k_aug, v_aug


def fa_intervention_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    scaling=None,
    dropout=0.0,
    sliding_window=None,
    attention_logits_intervention_vector=None,
    **kwargs,
):
    """Drop-in replacement for `eager_attention_forward` when intervention is active.

    Computes DySCO's biased attention via the augmented-dimension trick using the
    unmodified FlashAttention-2 kernel. Call signature mirrors
    `eager_attention_forward` so it can be slotted into the same dispatch point.

    Returns:
        (attn_output, None) where attn_output is (B, L_q, H_q, d) -- already in
        FA's (B, L, H, D) layout, which is what the caller's reshape expects
        (no transpose-back needed, unlike the eager path).
    """
    # Validate API usage before importing flash_attn, so misuse is caught even
    # in environments where flash_attn is not installed.
    assert attention_logits_intervention_vector is not None, (
        "fa_intervention_attention_forward requires an intervention vector; "
        "vanilla attention should use the standard FA dispatch path."
    )
    head_dim = query.shape[-1]
    expected_scale = 1.0 / math.sqrt(head_dim)
    assert scaling is not None and abs(scaling - expected_scale) < 1e-6, (
        f"softmax scaling must be 1/sqrt(d)={expected_scale:.6f} for the augmented-"
        f"dimension trick, got {scaling}. (FA's default 1/sqrt(d+1) breaks the math.)"
    )

    # Lazy import so environments without flash_attn can still run the eager path.
    from flash_attn import flash_attn_func

    q_aug, k_aug, v_aug = build_augmented_qk(
        query, key, value, attention_logits_intervention_vector, head_dim
    )

    # FA expects (B, L, H, D); model tensors are (B, H, L, D). FA handles GQA
    # natively (H_kv < H_q), so no repeat_kv needed.
    window = (sliding_window - 1, 0) if sliding_window else (-1, -1)
    out_aug = flash_attn_func(
        q_aug.transpose(1, 2),
        k_aug.transpose(1, 2),
        v_aug.transpose(1, 2),
        dropout_p=dropout,
        softmax_scale=expected_scale,  # explicit 1/sqrt(d), NOT FA's default 1/sqrt(d+1)
        causal=True,
        window_size=window,
    )  # (B, L_q, H_q, d+1)

    # Slice off the zero-valued augmented channel. Output is already (B, L_q, H_q, d).
    attn_output = out_aug[..., :head_dim].contiguous()
    return attn_output, None


# ============================================================================
# Speculative-pass attention: vanilla FA + per-layer QRHead snapshot
# ============================================================================
#
# DySCO's speculative pass needs the attention weights of the pre-detected
# QRHeads (a small set of (layer, head) pairs) to score context-token
# importance. Using `output_attentions=True` would force eager attention and
# materialize the FULL (B, H, L_q, L_k) attention matrix for ALL heads of
# every forwarded layer -- expensive and wasteful, since only the QRHead
# vectors are read downstream.
#
# Instead:
#   1. Vanilla FlashAttention computes the attention output (avoids the 4x
#      repeat_kv expansion + the fp32 full-attn buffer materialization).
#   2. Separately, for ONLY the QRHeads of the current layer, we compute
#      softmax(Q[h] . K[kv(h)]^T / sqrt(d)) and stash it in a shared dict
#      keyed by (layer_idx, head_idx). Cost: O(|QRHeads| * L_k * d) per step
#      across all selected layers (~16 heads total) -- negligible.


def materialize_qrhead_attn(
    query, key_full, head_dim, qrhead_query_indices, num_kv_groups,
    layer_idx, snapshot_buffer,
):
    """Compute softmax(Q[h] . K[kv(h)]^T / sqrt(d)) for the listed query heads.

    Args:
        query:                (B, H_q,  L_q=1, d)   only the new token's query
        key_full:             (B, H_kv, L_k,   d)   cat(cache_K, K_new) or full cache after update
        head_dim:             d
        qrhead_query_indices: list[int]  query head indices in THIS layer
        num_kv_groups:        H_q // H_kv  (GQA ratio)
        layer_idx:            int  for snapshot dict key
        snapshot_buffer:      dict mutated in place: {(layer_idx, head_idx): (B, L_k) attn vector}

    Causal mask is trivial at decode (L_q=1, query attends to all keys
    including itself), so no masking is applied. Softmax runs in fp32 then
    casts back to query dtype, matching the eager path's precision behavior.
    """
    if not qrhead_query_indices:
        return
    scale = 1.0 / math.sqrt(head_dim)
    for h in qrhead_query_indices:
        h_kv = h // num_kv_groups
        q_h = query[:, h, 0, :]                                    # (B, d)
        k_h = key_full[:, h_kv, :, :]                              # (B, L_k, d)
        logits = torch.einsum("bd,bld->bl", q_h, k_h) * scale      # (B, L_k)
        snapshot_buffer[(layer_idx, h)] = torch.softmax(
            logits, dim=-1, dtype=torch.float32
        ).to(query.dtype)


def fa_spec_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    scaling=None,
    dropout=0.0,
    sliding_window=None,
    qrhead_indices_in_layer=None,
    qrhead_snapshot_buffer=None,
    **kwargs,
):
    """Vanilla FlashAttention + per-layer QRHead snapshot.

    Drop-in replacement for `eager_attention_forward` when the caller wants
    QRHead attention vectors but NOT the full all-head attention matrix. The
    main attention output flows through normal FA; the snapshot is written
    side-band into `qrhead_snapshot_buffer`.

    Call expects `query`/`key`/`value` already in (B, H, L, D) torch layout
    and `key`/`value` to be the FULL post-cat (or post-update) tensors
    covering the entire context including the current token's K/V.

    Returns (attn_output, None) where attn_output is (B, L_q, H_q, d) in
    FA's (B, L, H, D) layout -- ready for the caller's reshape.
    """
    head_dim = query.shape[-1]
    expected_scale = 1.0 / math.sqrt(head_dim)
    assert scaling is not None and abs(scaling - expected_scale) < 1e-6, (
        f"scaling must be 1/sqrt(d)={expected_scale:.6f}, got {scaling}"
    )

    # Lazy import: callers that don't use FA shouldn't need the package installed.
    from flash_attn import flash_attn_func

    window = (sliding_window - 1, 0) if sliding_window else (-1, -1)
    out = flash_attn_func(
        query.transpose(1, 2),    # (B, L_q, H_q,  d)
        key.transpose(1, 2),      # (B, L_k, H_kv, d)  -- FA handles GQA natively
        value.transpose(1, 2),
        dropout_p=dropout,
        softmax_scale=expected_scale,
        causal=True,
        window_size=window,
    )                              # (B, L_q, H_q, d)
    attn_output = out.contiguous()

    # Snapshot the QRHead attention vectors for THIS layer (if any).
    if qrhead_indices_in_layer and qrhead_snapshot_buffer is not None:
        materialize_qrhead_attn(
            query, key, head_dim,
            qrhead_indices_in_layer,
            module.num_key_value_groups,
            module.layer_idx,
            qrhead_snapshot_buffer,
        )

    return attn_output, None
