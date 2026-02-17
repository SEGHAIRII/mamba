# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

from typing import Optional

import math
from packaging import version

import torch
import torch.nn.functional as F
from torch import Tensor
from mamba_ssm.utils.torch import custom_bwd, custom_fwd

import triton
import triton.language as tl

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.cpp_functions import causal_conv1d_fwd_function, causal_conv1d_bwd_function, causal_conv1d_update_function
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None
    causal_conv1d_update_function = None

from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_cumsum_bwd, _chunk_cumsum_A_fwd
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd, _chunk_state_bwd_db, _chunk_state_dr_fwd, _chunk_state_bwd_db_dr
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_ddAcs_stable
from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state, chunk_state_ref
from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state_varlen
from mamba_ssm.ops.triton.ssd_chunk_state import _precompute_mask
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd, _state_passing_bwd, _state_passing_dr_fwd, _state_passing_dr_bwd
from mamba_ssm.ops.triton.ssd_state_passing import state_passing, state_passing_ref
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd, _chunk_scan_bwd_dz, _chunk_scan_bwd_dstates, _chunk_scan_dr_fwd, _chunk_scan_bwd_dr_dz, _chunk_scan_bwd_dr_dstates
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dC, _chunk_scan_bwd_dcb, _chunk_scan_bwd_dC_dr, _chunk_scan_bwd_dcb_dr
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_ddAcs_stable, _chunk_scan_bwd_ddAcs_stable_dr
from mamba_ssm.ops.triton.ssd_chunk_scan import chunk_scan, chunk_scan_ref, _chunk_scan_chunk_state_bwd_dx_dr
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_ddAcs_prev
from mamba_ssm.ops.triton.inhibition_gate import _inhibition_gate_fwd, _inhibition_gate_bwd
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn, _layer_norm_fwd, _layer_norm_bwd
from mamba_ssm.ops.triton.k_activations import _swiglu_fwd, _swiglu_bwd, _reglu_fwd, _reglu_bwd

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')




def calculate_activation_sparsity(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Calculate sparsity as the proportion of near-zero activations along the hidden dimension.
    Averages over batch and sequence dimensions.
    
    Args:
        hidden_states: Tensor of shape [batch, seq_len, hidden_dim]
    
    Returns:
        sparsity: Scalar tensor representing average sparsity across batch and sequence
    """
    # Calculate sparsity along hidden dimension: [batch, seq_len]
    sparsity_per_position = (hidden_states == 0).float().mean(dim=-1)
    # Average over batch and sequence: scalar
    sparsity = sparsity_per_position.mean()
    return sparsity * 100

def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


def rearrange_and_update_stride(tensor, pattern=None, dim=2):
    # ensure tensor.stride(dim) is a multiple of eight after rearranging according to pattern,
    # if not call contiguous(), rearrange only if pattern is not None
    tensor_rearranged = rearrange(tensor, pattern) if pattern is not None else tensor
    return tensor_rearranged.contiguous() if tensor_rearranged.stride(dim) % 8 != 0 else tensor_rearranged


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, cb_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, D_ptr,
    b_ptr, dstates_ptr,
    dx_ptr, ddt_ptr, dD_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_D_head,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    if not HAS_SEQ_IDX:
        # scale = tl.exp(dA_cs_last - dA_cs_m)
        scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        # scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)), 0.0)
    # Might be faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    # However, we're getting error with the Triton compiler 2.1.0 for that code path:
    # Unexpected mma -> mma layout conversion
    # Triton 2.2.0 fixes this
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_dstate[:, None] * stride_dstates_dstate)
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    # x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    # x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    # dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    # dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    # ddt = tl.sum(acc * x, axis=1) * dt_m
    # ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    # tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit
    K_MIN = pid_m * BLOCK_SIZE_M
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    dA_cumsum_ptrs += K_MIN * stride_dA_cs_csize
    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        # For some reason setting mask to (offs_m[:, None] < chunk_size_limit) is much slower
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < K_MAX - k, other=0.0).to(tl.float32)
        # cb *= tl.exp(dA_cs_k[None, :] - dA_cs_m[:, None])
        cb *= tl.exp(tl.minimum((dA_cs_k[None, :] - dA_cs_m[:, None]), 0.0))
        # If we don't have the (k + offs_k[None, :] < K_MAX) mask, for indices outside this range,
        # we might have dA_cs_m = 0.0 and dA_cs_k very negative, and tl.exp will return inf.
        # Multiplying with cb, which is 0.0 outside the range, will make the result NaN.
        # This will cause NaN in acc, and hence NaN in dx and ddt.
        mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < K_MAX)
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
        dout_res = tl.load(dout_res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


#dr
@triton.jit
def _chunk_scan_through_mask_bwd_kernel(
    # Pointers to matrices
    out_ptr, M_ptr, dout_ptr, dMask_ptr,
    # Matrix dimensions  
    batch, seqlen, nheads, headdim,
    # Strides for out (y)
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    # Strides for M (precomputed mask)
    stride_M_batch, stride_M_seqlen, stride_M_head, stride_M_hdim,
    # Strides for dout
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,    
    # Strides for dMask output
    stride_dMask_batch, stride_dMask_seqlen, stride_dMask_head, stride_dMask_hdim,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Backward through output mask: out = y * M
    
    Forward: out = y * M
    Backward: 
        dy = dout * M (gradient to propagate to SSM)
        dM = dout * y = dout * (out / M) where M != 0
           For M = 0, y doesn't contribute to out, so dM = 0 there
           Since out = y * M, when M = 1, y = out, so dM = dout * out / M = dout * out
           When M = 0, out = 0, dM should be dout * y but y is irrelevant since M=0
           
    IMPORTANT: We need y (pre-mask output) but we only have out = y * M
    Since M is binary (0 or 1): when M=1, y = out; when M=0, y is unknown but dM=0 anyway
    So: dM = dout * out (where M=1) and dM = 0 (where M=0)
    Equivalently: dM = dout * out * M / M = dout * out (for M=1), dM = 0 (for M=0)
    But simpler: dM = dout * out (works because out=0 when M=0)
    
    Wait, let me reconsider:
    - out = y * M
    - When M = 1: out = y, so dM = dout * y = dout * out ✓
    - When M = 0: out = 0, dM = dout * y but this gradient doesn't flow anyway
      (the mask completely blocks the output, so dM at M=0 doesn't affect loss)
      
    Actually dM = dout * y. Since we don't store y separately:
    - When M = 1: y = out, so dM = dout * out ✓
    - When M = 0: y ≠ out, but dM = dout * y. However, since M=0 is hard gating,
      we should still compute the STE gradient through y, not 0.
      
    The issue: we need to pass y (unmasked output) through, or recover it.
    Since out = y * M and M is binary: y = out / M (undefined for M=0)
    
    For STE, we actually want the gradient as if the mask wasn't there.
    So dM should be dout * y (the unmasked SSM output).
    
    SOLUTION: We need y, not out. But y isn't stored separately.
    For now, approximate: dM = dout * (out / M) where M=1, else dM = 0
    This is equivalent to: dM = dout * out (since out=y when M=1, out=0 when M=0)
    """
    # Program IDs - restructured to avoid exceeding CUDA grid limits
    pid_blh = tl.program_id(axis=0)  # Merged batch*seqlen*nheads
    pid_n = tl.program_id(axis=1)    # Block in headdim
    
    # Decompose the merged index
    pid_b = pid_blh // (seqlen * nheads)
    pid_lh = pid_blh % (seqlen * nheads)
    pid_l = pid_lh // nheads
    pid_h = pid_lh % nheads
    
    # Offsets within headdim
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < headdim
    
    # Pointer setup
    out_ptr += pid_b * stride_out_batch + pid_l * stride_out_seqlen + pid_h * stride_out_head
    M_ptr += pid_b * stride_M_batch + pid_l * stride_M_seqlen + pid_h * stride_M_head
    dout_ptr += pid_b * stride_dout_batch + pid_l * stride_dout_seqlen + pid_h * stride_dout_head
    dMask_ptr += pid_b * stride_dMask_batch + pid_l * stride_dMask_seqlen + pid_h * stride_dMask_head
    
    # Load headdim data
    out = tl.load(out_ptr + offs_n * stride_out_hdim, mask=mask, other=0.0).to(tl.float32)
    M = tl.load(M_ptr + offs_n * stride_M_hdim, mask=mask, other=0.0).to(tl.float32)
    dout_val = tl.load(dout_ptr + offs_n * stride_dout_hdim, mask=mask, other=0.0).to(tl.float32)
    
    # Compute gradients:
    # out = y * M (forward), where y is pre-mask SSM output
    # dy = dout * M (correct - gradient propagates where mask is 1)
    # dM = dout * y (correct - but we don't have y directly)
    #
    # Since M is binary (0 or 1) and out = y * M:
    # - When M = 1: y = out, so dM = dout * out ✓
    # - When M = 0: out = 0, y is unknown, but gradient doesn't flow anyway
    #   because the hard mask blocks the path
    #
    # dM = dout * out is correct for the positions where M=1.
    # For M=0, dM doesn't matter for the forward loss (mask blocked it).
    # But for STE, we want gradient to flow. Since out=0 when M=0,
    # dM = dout * out = 0 there, which is consistent with STE through hard mask.
    dMask = dout_val * out  # dM = dout * y ≈ dout * out (valid since out=y when M=1)
    
    # new_dout = gradient to propagate backward through mask (M * dout)
    # This is the gradient w.r.t. the SSM output before final masking
    new_dout = M * dout_val
    
    # Store results
    tl.store(dMask_ptr + offs_n * stride_dMask_hdim, dMask, mask=mask)
    tl.store(dout_ptr + offs_n * stride_dout_hdim, new_dout, mask=mask)
    
    

#dr
def _chunk_scan_through_mask_bwd(out, M, dout, dMask):
    """
    Backward through the output mask. Uses precomputed M instead of E, I, E_mean.
    """
    batch, seqlen, nheads, headdim = out.shape
    assert M.shape == (batch, seqlen, nheads, headdim)
    assert dout.shape == out.shape
    assert dMask.shape == (batch, seqlen, nheads, headdim)
    # Grid: (batch * seqlen * nheads, headdim_blocks) to avoid exceeding CUDA limits
    grid_dx = lambda META: (batch * seqlen * nheads, triton.cdiv(headdim, META['BLOCK_SIZE_N']))
    with torch.cuda.device(out.device.index):
        _chunk_scan_through_mask_bwd_kernel[grid_dx](
            out, M, dout, dMask,
            # Matrix dimensions
            batch, seqlen, nheads, headdim,
            # Strides
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dMask.stride(0), dMask.stride(1), dMask.stride(2), dMask.stride(3),
            BLOCK_SIZE_N=64
        )
        
    return dout, dMask
     



def _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=None, seq_idx=None, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                        batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x, CB, dout, dt, dA_cumsum, seq_idx, D, B, dstates, dx, ddt, dD,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(-1), CB.stride(-2),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            D.stride(0) if D is not None else 0,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=TRITON_22
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return dx, ddt.to(dtype=dt.dtype), dD


def _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), output_activation = None, threshold= 0.0):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    # # (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, nheads, chunk_size, chunk_size)
    # dA_cumsum_tmp0, dt_tmp0 = _chunk_cumsum_fwd(dt[:, :147], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp1, dt_tmp1 = _chunk_cumsum_fwd(dt[:, 147:], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp2, dt_tmp2 = _chunk_cumsum_fwd(dt[:, 147:256], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    # states_tmp0 = _chunk_state_fwd(B[:, :147], x[:, :147], dt_tmp0, dA_cumsum_tmp0, states_in_fp32=True)
    # states_tmp1 = _chunk_state_fwd(B[:, 147:], x[:, 147:], dt_tmp1, dA_cumsum_tmp1, states_in_fp32=True)
    # states_tmp2 = _chunk_state_fwd(B[:, 147:256], x[:, 147:256], dt_tmp2, dA_cumsum_tmp2, states_in_fp32=True)
    states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                              initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                              seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    # states_tmp0 = rearrange(_state_passing_fwd(rearrange(states_tmp0, "... p n -> ... (p n)"), dA_cumsum_tmp0[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    # states_tmp1 = rearrange(_state_passing_fwd(rearrange(states_tmp1, "... p n -> ... (p n)"), dA_cumsum_tmp1[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx, output_activation=output_activation, threshold=threshold)
    if cu_seqlens is None:
        return out, out_x, dt, dA_cumsum, states, final_states
    else:
        assert batch == 1, "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        varlen_states = chunk_state_varlen(B.squeeze(0), x.squeeze(0), dt.squeeze(0), dA_cumsum.squeeze(0),
                                           cu_seqlens, states.squeeze(0))
        return out, out_x, dt, dA_cumsum, states, final_states, varlen_states


def _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, chunk_size, D=None, z=None,
                                   dt_bias=None, initial_states=None, dfinal_states=None, seq_idx=None, dt_softplus=False,
                                   dt_limit=(0.0, float("inf")),
                                   dx=None, ddt=None, dB=None, dC=None, dz=None, recompute_output=False, output_activation = None, threshold = 0.0):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    _, _, ngroups, dstate = B.shape
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert out.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt)
    # TD: For some reason Triton (2.1.0 and 2.2.0) errors with
    # "[CUDA]: invalid device context" (e.g. during varlne test), and cloning makes it work. Idk why.
    dt_in = dt.clone()
    dA_cumsum, dt = _chunk_cumsum_fwd(dt_in, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus,
                                      dt_limit=dt_limit)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, _ = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                   initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                   seq_idx=seq_idx, chunk_size=chunk_size)
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(x, z, out, dout, chunk_size=chunk_size, has_ddAcs=False, D=D, dz=dz, recompute_output=recompute_output, output_activation=output_activation, threshold=threshold)
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        if output_activation == 'relu':
            dout = dout * (out > threshold).to(dout.dtype)
        outz = out
        
    
    dstates = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=seq_idx, dtype=states.dtype)
    # dstates has length nchunks, containing the gradient to initial states at index 0 and
    # gradient to the states of chunk (nchunks - 2) at index (nchunks - 1)
    # Do computation in fp32 but convert dstates and states to fp16/bf16 since dstates and states
    # will be used in matmul in the next kernels.
    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        rearrange(dstates, "... p n -> ... (p n)"),
        dfinal_states=rearrange(dfinal_states, "... p n -> ... (p n)") if dfinal_states is not None else None,
        seq_idx=seq_idx,
        has_initial_states=initial_states is not None,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    # dstates has length nchunks, containing the gradient to states of chunk 0 at index 0 and
    # gradient to the final states at index (nchunks - 1)
    # states has length nchunks, containing the initial states at index 0 and the state for chunk (nchunks - 2) at index (nchunks - 1)
    # The final states is not stored.
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)
    dinitial_states = rearrange(dinitial_states, "... (p n) -> ... p n", n=dstate) if dinitial_states is not None else None
    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=D, seq_idx=seq_idx, dx=dx)
    # dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, ngroups=ngroups)
    dB, ddA_next = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups)
    # dC = _chunk_scan_bwd_dC(states[:, :-1].to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    dC, ddA_cumsum_prev = _chunk_scan_bwd_dC(states.to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups)
    # Computing ddA with the dcb kernel is much slower, so we're not using it for now
    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    # dCB, ddA_tmp = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, CB=CB, ngroups=ngroups)
    dCB = dCB.to(CB.dtype)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
    _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_given)
    # If we have z, then dout_x is recomputed in fp32 so dD = (dout_x * x).sum() is more accurate
    # than dD_from_x = (dout_x * x).sum() where dout_x is in fp16/bf16
    if z is None:
        dD = dD_from_x
    # Formula for ddA_cumsum, assuming out is the output of the forward pass before adding x * D.
    # ddA_cumsum = torch.einsum("bclhp,bclhp->bhcl", out.float(), dout.float()) - ddt * dt
    # However, this is numerically unstable: when we do the reverse cumsum on ddA_cumsum, there might
    # be a lot of underflow.

    # This is already done as part of bwd_dC kernel
    # ddA_cumsum_prev = _chunk_scan_bwd_ddAcs_prev(states[:, :-1], C, dout, dA_cumsum, seq_idx=seq_idx)
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    # This is already done as part of bwd_dB kernel
    # ddA_next = _chunk_state_bwd_ddAcs_stable(B, x, dt, dA_cumsum, dstates, seq_idx=seq_idx)
    # We don't need to pass in seq_idx because CB also zeros out entries where seq_idx[i] != seq_idx[j]
    ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB)
    ddA += ddA_next + ddA_prev

    ddt_given, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt, dt_in, A, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit, ddt=ddt_given)

    # These 2 lines are just to test ddt and dA being computed by old code
    # _, dA = selective_scan_bwd(dout, x, dt, A, B, C, D=D.float(), z=z)
    # ddt_given.copy_(ddt)

    return_vals = (dx, ddt_given, dA, dB_given, dC_given, dD, dz, ddt_bias, dinitial_states)
    return return_vals if not recompute_output else (*return_vals, outz)



#dr
def _dr_ssm_chunk_scan_combined_fwd(x, A, B, C, E, I, chunk_size, D=None, z=None, initial_states=None, seq_idx=None, cu_seqlens=None, output_activation=None, threshold=0.0):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert E.shape == (batch, seqlen, nheads, headdim)
    assert I.shape == (batch, seqlen, nheads, headdim)
    assert A.shape == (batch, seqlen, nheads, headdim)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if E.stride(-1) != 1 and E.stride(1) != 1:
        E = E.contiguous()
    if I.stride(-1) != 1 and I.stride(1) != 1:
        I = I.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
        
    print(f"[DEBUG FWD] Starting forward pass: batch={batch}, seqlen={seqlen}, nheads={nheads}, headdim={headdim}", flush=True)
    
    # OPTIMIZATION: Precompute mask, x_masked, sigmoid_delta, E_mean all at once
    # This eliminates redundant computation in all downstream kernels
    x_masked, M, sigmoid_delta, E_mean, sigma_squared_sum = _precompute_mask(E, I, x)
    print("[DEBUG FWD] After _precompute_mask", flush=True)
    
    # Calculate the cumulative sum of the negative decay A over chunks
    dA_cumsum = _chunk_cumsum_A_fwd(A, chunk_size)
    print("[DEBUG FWD] After _chunk_cumsum_A_fwd", flush=True)
    
    # Calculate the final decayed chunks using precomputed x_masked
    states = _chunk_state_dr_fwd(B, x_masked, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    print("[DEBUG FWD] After _chunk_state_dr_fwd", flush=True)
    
    # Pass the final decayed states between chunks
    states, final_states = _state_passing_dr_fwd(
        rearrange(states, "... p n -> ... (p n)"), 
        dA_cumsum[:, :, :, -1, :],
        initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
        seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype
    )
    print("[DEBUG FWD] After _state_passing_dr_fwd", flush=True)
    
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]

    # Calculate the chunked BMM of C and B
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    print("[DEBUG FWD] After _bmm_chunk_fwd", flush=True)
    
    # Calculate the real ssm output using x_masked and M
    # TODO: _chunk_scan_dr_fwd needs to be updated to use x_masked and M
    out, out_x = _chunk_scan_dr_fwd(CB, x_masked, dA_cumsum, C, M, states, D=D, z=z, seq_idx=seq_idx, output_activation=output_activation, threshold=threshold)
    print("[DEBUG FWD] After _chunk_scan_dr_fwd - Forward pass complete!", flush=True)
    
    if cu_seqlens is None:
        # Return sigma_squared_sum for regularization loss computation
        return out, out_x, sigma_squared_sum, dA_cumsum, states, final_states, x_masked, M, sigmoid_delta, E_mean
    else:
        assert batch == 1, "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        varlen_states = chunk_state_varlen(B.squeeze(0), x.squeeze(0), dt.squeeze(0), dA_cumsum.squeeze(0),
                                           cu_seqlens, states.squeeze(0))
        return out, out_x, sigma_squared_sum, dA_cumsum, states, final_states, varlen_states, x_masked, M, sigmoid_delta, E_mean

#dr
def _dr_ssm_chunk_scan_combined_bwd(dout, x, A, B, C, E, I, out, chunk_size, D=None, z=None,
                                   initial_states=None, dfinal_states=None, seq_idx=None,
                                   dx=None, dB=None, dC=None, dE=None, dI=None, dA=None, dz=None, recompute_output=False, output_activation = None, threshold = 0.0):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    _, _, ngroups, dstate = B.shape
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert A.shape == (batch, seqlen, nheads, headdim)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert E.shape == (batch, seqlen, nheads, headdim)
    assert I.shape == (batch, seqlen, nheads, headdim)
    assert out.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if dA is not None:
        assert dA.shape == (batch, seqlen, nheads, headdim)
        dA_given = dA
    else:
        dA_given = torch.empty_like(A)
    if dE is not None:
        assert dE.shape == E.shape
        dE_given = dE
    else:
        dE_given = torch.empty_like(E)
    if dI is not None:
        assert dI.shape == I.shape
        dI_given = dI
    else:
        dI_given = torch.empty_like(I)
        
        
    # Precompute mask and masked input - same as forward pass
    x_masked, M, sigmoid_delta, E_mean, _ = _precompute_mask(E, I, x)
    
    ddA_next = torch.empty(batch, seqlen, nheads, headdim, 
                          device=x.device, dtype=torch.float32)
    ddA_prev = torch.empty(batch, seqlen, nheads, headdim, 
                          device=x.device, dtype=torch.float32)
    ddA_stable = torch.empty(batch, seqlen, nheads, headdim, 
                          device=x.device, dtype=torch.float32)

    print("[DEBUG BWD] Starting backward pass", flush=True)
    dA_cumsum = _chunk_cumsum_A_fwd(A, chunk_size)
    print("[DEBUG BWD] After _chunk_cumsum_A_fwd", flush=True)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    print("[DEBUG BWD] After _bmm_chunk_fwd", flush=True)
    # Use x_masked from precompute - now matches the forward signature
    states = _chunk_state_dr_fwd(B, x_masked, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    print("[DEBUG BWD] After _chunk_state_dr_fwd", flush=True)
    states, _ = _state_passing_dr_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1, :],
                                   initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                   seq_idx=seq_idx, chunk_size=chunk_size)
    print("[DEBUG BWD] After _state_passing_dr_fwd", flush=True)
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(x, z, out, dout, chunk_size=chunk_size, has_ddAcs=False, D=D, dz=dz, recompute_output=recompute_output, output_activation=output_activation, threshold=threshold)
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        dD = None  # Will be computed if D is not None
        if output_activation == 'relu':
            dout = dout * (out > threshold).to(dout.dtype)
        outz = out
        
        
    # beginning of backward
    # Initialize dMask for accumulating mask gradients
    dMask = torch.empty_like(E)
    print("[DEBUG BWD] Before _chunk_scan_through_mask_bwd", flush=True)
    dout, dMask = _chunk_scan_through_mask_bwd(out, M, dout, dMask)
    print("[DEBUG BWD] After _chunk_scan_through_mask_bwd", flush=True)
    dstates = _chunk_scan_bwd_dr_dstates(C, dA_cumsum, dout, seq_idx=seq_idx, dtype=states.dtype)     # recheck
    print("[DEBUG BWD] After _chunk_scan_bwd_dr_dstates", flush=True)

    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_dr_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1, :],
        rearrange(dstates, "... p n -> ... (p n)"),
        dfinal_states=rearrange(dfinal_states, "... p n -> ... (p n)") if dfinal_states is not None else None,
        seq_idx=seq_idx,
        has_initial_states=initial_states is not None,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    print("[DEBUG BWD] After _state_passing_dr_bwd", flush=True)
    
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)
    dinitial_states = rearrange(dinitial_states, "... (p n) -> ... p n", n=dstate) if dinitial_states is not None else None
 
    print("[DEBUG BWD] Before _chunk_scan_chunk_state_bwd_dx_dr", flush=True)
    dx =  _chunk_scan_chunk_state_bwd_dx_dr(x_masked, M, dout, dstates, B, CB, dA_cumsum, seq_idx=seq_idx, dx=dx, dMask=dMask)    
    print("[DEBUG BWD] After _chunk_scan_chunk_state_bwd_dx_dr", flush=True)
    dB, ddA_next = _chunk_state_bwd_db_dr(x_masked=x_masked,M=M,dstates=dstates,B=B,dA_cumsum=dA_cumsum,seq_idx=seq_idx,dB=None,ddA_next=ddA_next,has_ddA=True)
    print("[DEBUG BWD] After _chunk_state_bwd_db_dr", flush=True)
    dC, ddA_prev = _chunk_scan_bwd_dC_dr(dout=dout,M=M,states=states,C=C,dA_cumsum=dA_cumsum,seq_idx=seq_idx,dC=None,ddA_prev=ddA_prev,has_ddA=True)
    print("[DEBUG BWD] After _chunk_scan_bwd_dC_dr", flush=True)

    dCB = _chunk_scan_bwd_dcb_dr(x_masked, M, dout, dA_cumsum, ngroups=ngroups)
    print("[DEBUG BWD] After _chunk_scan_bwd_dcb_dr", flush=True)

    dCB = dCB.to(CB.dtype)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
    _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_given)
    print("[DEBUG BWD] After _bmm_chunk_bwd", flush=True)

    # dD is already computed above (either from _chunk_scan_bwd_dz if z is not None, or None if z is None)
    # If D is provided but z is None, dD should be computed from dout * x
    if z is None and D is not None:
        dD = torch.einsum('blhp,blhp->hp', dout.float(), x.float()).to(x.dtype)
        if D.dim() == 1:
            dD = dD.sum(dim=-1)

    # ddA_chunk_cumsum: (batch, nheads, nchunks, headdim)
    # Expand to (batch, nheads, seqlen, headdim) and transpose to (batch, seqlen, nheads, headdim)
    ddA_chunk_expanded = torch.repeat_interleave(
        ddA_chunk_cumsum, chunk_size, dim=2
    )[:, :, :seqlen, :]  # (B, H, L, P)

    ddA_stable = _chunk_scan_bwd_ddAcs_stable_dr(x_masked, M, dout, CB, dA_cumsum)
    ddA_total = ddA_chunk_expanded.transpose(1, 2) + ddA_next + ddA_prev + ddA_stable  # All (B, L, H, P)
    dA = torch.flip(torch.cumsum(torch.flip(ddA_total, dims=[1]), dim=1), dims=[1])
    
    # STE gradient: use precomputed sigmoid_delta for efficiency
    # delta = E - I - E_mean, sigma = sigmoid(delta), M = (sigma > 0.5)
    # d(loss)/dE = d(loss)/dM * d(sigma)/d(delta) * d(delta)/dE
    # d(delta)/dE = 1 - 1/P (due to mean subtraction), d(delta)/dI = -1
    # sigmoid_delta is already computed by _precompute_mask above
    dsigma = dMask * sigmoid_delta * (1 - sigmoid_delta)
    dE_given = dsigma - dsigma.mean(dim=-1, keepdim=True)
    dI_given = -dsigma

    return_vals = (dx, dA, dB_given, dC_given, dD, dz, dE_given, dI_given, dinitial_states)
    return return_vals if not recompute_output else (*return_vals, outz)


def selective_scan_bwd(dout, x, dt, A, B, C, D=None, z=None):
    """
    Argument:
        dout: (batch, seqlen, nheads, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size) or (batch, nheads, headdim, nchunks, chunk_size)
        A: (nheads) or (dim, dstate)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    import selective_scan

    batch, seqlen, nheads, headdim = x.shape
    chunk_size = dt.shape[-1]
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    x = rearrange(x, "b l h p -> b (h p) l")
    squeeze_dt = dt.dim() == 4
    if dt.dim() == 4:
        dt = repeat(dt, "b h c l -> b h p c l", p=headdim)
    dt = rearrange(dt, "b h p c l -> b (h p) (c l)", p=headdim)
    squeeze_A = A.dim() == 1
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    B = rearrange(B, "b l g n -> b g n l")
    C = rearrange(C, "b l g n -> b g n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")

    if x.stride(-1) != 1:
        x = x.contiguous()
    if dt.stride(-1) != 1:
        dt = dt.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    _, intermediate, *rest = selective_scan.fwd(x, dt.to(dtype=x.dtype), A, B, C, D, z, None, False)
    if z is not None:
        out = rest[0]
    else:
        out = None

    dout = rearrange(dout, "b l h p -> b (h p) l")

    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
    # backward of selective_scan with the backward of chunk).
    # Here we just pass in None and dz will be allocated in the C++ code.
    _, ddt, dA, *rest = selective_scan.bwd(
        x, dt.to(dtype=x.dtype), A, B, C, D, z, None, dout, intermediate, out, None, False,
        False  # option to recompute out_z, not used here
    )
    ddt = rearrange(ddt, "b (h p) (c l) -> b h p c l", p=headdim, l=chunk_size)
    if squeeze_dt:
        ddt = ddt.float().sum(dim=2)
    if squeeze_A:
        dA = rearrange(dA, "(h p) n -> h p n", p=headdim).sum(dim=(1, 2))
    return ddt, dA


class MambaChunkScanCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, return_varlen_states=False, return_activation_sparsity=False, output_activation = None, threshold = 0.0):
        ctx.dt_dtype = dt.dtype
        if not return_varlen_states:
            cu_seqlens = None
        else:
            assert cu_seqlens is not None, "cu_seqlens must be provided if return_varlen_states is True"
        out, out_x, dt_out, dA_cumsum, states, final_states, *rest = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, dt_limit=dt_limit, output_activation=output_activation, threshold=threshold)
        if return_activation_sparsity:
            if z is not None:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)")) 
            else:
                activation_sparsity = calculate_activation_sparsity(rearrange(out, "b s h p -> b s (h p)"))
        ctx.save_for_backward(out if z is None else out_x, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx)
        ctx.dt_softplus = dt_softplus
        ctx.chunk_size = chunk_size
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.return_varlen_states = return_varlen_states
        if not return_varlen_states:
            if not return_activation_sparsity:
                return out if not return_final_states else (out, final_states)
            else:
                if not return_final_states:
                    return (out, activation_sparsity)
                else:
                    return (out, final_states, activation_sparsity)
        else:
            varlen_states = rest[0]
            if not return_activation_sparsity:
                return (out, varlen_states) if not return_final_states else (out, final_states, varlen_states)
            else:
                if not return_final_states:
                    return (out, varlen_states, activation_sparsity)
                else:
                    return (out, final_states, varlen_states, activation_sparsity)      
    @staticmethod
    def backward(ctx, dout, *args):
        out, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx = ctx.saved_tensors
        assert not ctx.return_varlen_states, "return_varlen_states is not supported in backward"
        dfinal_states = args[0] if ctx.return_final_states else None
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit)
        return dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, dinitial_states, None, None, None, None, None, None, None


def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, return_varlen_states=False, return_activation_sparsity = False, output_activation = None, threshold = 0.0):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    return MambaChunkScanCombinedFn.apply(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, dt_limit, return_final_states, return_varlen_states, return_activation_sparsity, output_activation, threshold)


def mamba_chunk_scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # We want high precision for this before cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    # 1. Compute the state for each chunk
    states = chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True)
    # 2. Pass the state to all the chunks by weighted cumsum.
    states = rearrange(state_passing(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                       "... (p n) -> ... p n", n=dstate)
    # 3. Compute the output for each chunk
    out = chunk_scan(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out


def ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # We want high precision for this before cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    # 1. Compute the state for each chunk
    states = chunk_state_ref(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)
    # 2. Pass the state to all the chunks by weighted cumsum.
    # state_passing_ref is much less numerically stable
    states = rearrange(state_passing_ref(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                       "... (p n) -> ... p n", n=dstate)
    states = states.to(states_dtype)
    # 3. Compute the output for each chunk
    out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out


def ssd_selective_scan(x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads) or (batch, seqlen, nheads, headdim)
        A: (nheads) or (dim, dstate)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,) or (nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    x = rearrange(x, "b l h p -> b (h p) l")
    if dt.dim() == 3:
        dt = repeat(dt, "b l h -> b l h p", p=headdim)
    dt = rearrange(dt, "b l h p -> b (h p) l")
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    B = rearrange(B, "b l g n -> b g n l")
    C = rearrange(C, "b l g n -> b g n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")
    if dt_bias is not None:
        if dt_bias.dim() == 1:
            dt_bias = repeat(dt_bias, "h -> h p", p=headdim)
        dt_bias = rearrange(dt_bias, "h p -> (h p)")
    if dt_limit != (0.0, float("inf")):
        if dt_bias is not None:
            dt = dt + rearrange(dt_bias, "d -> d 1")
        if dt_softplus:
            dt = F.softplus(dt)
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1]).to(x.dtype)
        dt_bias = None
        dt_softplus = None
    out = selective_scan_fn(x, dt, A, B, C, D=D, z=z, delta_bias=dt_bias, delta_softplus=dt_softplus)
    return rearrange(out, "b (h p) l -> b l h p", p=headdim)


def mamba_conv1d_scan_ref(xBC, conv1d_weight, conv1d_bias, dt, A, chunk_size, D=None, z=None,
                          dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")),
                          activation="silu", headdim=None, ngroups=1):
    """
    Argument:
        xBC: (batch, seqlen, dim + 2 * ngroups * dstate) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt: (batch, seqlen, nheads) or (batch, seqlen, nheads, headdim)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, dim)
        dt_bias: (nheads) or (nheads, headdim)
        headdim: if D is 1D and z is None, headdim must be passed in
    Return:
        out: (batch, seqlen, dim)
    """
    batch, seqlen, nheads = dt.shape[:3]
    assert nheads % ngroups == 0
    if z is not None:
        dim = z.shape[-1]
        assert dim % nheads == 0
        headdim = dim // nheads
    else:
        if D.dim() == 1:
            assert headdim is not None
        else:
            headdim = D.shape[1]
        dim = nheads * headdim
    xBC = rearrange(causal_conv1d_fn(rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation=activation),
                    "b d s -> b s d")
    dstate = (xBC.shape[-1] - dim) // ngroups // 2
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
    out = ssd_selective_scan(x, dt.to(x.dtype), A, B, C, D=D.float(), z=z, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    return rearrange(out, "b s h p -> b s (h p)")


class MambaSplitConv1dScanCombinedFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
                rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None,
                ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold=0.0):
        # assert activation in [None, "silu", "swish", "relu"]
        if D.dim() == 1:
            assert headdim is not None
            nheads, = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        assert nheads % ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                                 conv1d_weight, conv1d_bias, seq_idx, None, None, True if activation in ["silu", "swish"] else None),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
        
        # C centered
        # C = C - C.mean(dim=-1, keepdim=True)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        if rmsnorm_weight is None:
            out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation, threshold=threshold)
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            out = rearrange(out, "b s h p -> b s (h p)")
            rstd = None
            if d_nonssm > 0:
                if activation in ["silu", "swish"]:
                    out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
                else:
                    out = torch.cat([_reglu_fwd(zx0), out], dim=-1)
        else:
            out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation, threshold=threshold)
            # reshape input data into 2D tensor
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            # rearranged_x = rearrange(out_x, "b s h p -> b s (h p)")
            # total_0 = (rearranged_x[-1] == 0).float().mean(-1).mean()
            # print(total_0)
            x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            rmsnorm_weight = rmsnorm_weight.contiguous()
            if d_nonssm == 0:
                out = None
            else:
                out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
                out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
                if activation == "silu" or activation == "swish":
                    out01[..., :d_nonssm] = _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
                elif activation == "relu":
                    out01[..., :d_nonssm] = _reglu_fwd(zx0, out=out01[..., :d_nonssm])
                    
            out, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=out,
                                           group_size=dim // ngroups,
                                           norm_before_gate=norm_before_gate, is_rms_norm=True)
            if d_nonssm == 0:
                out = rearrange(out, "(b s) d -> b s d", b=batch)
            else:
                out = out01
        ctx.outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
        if outproj_weight is not None:
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
                outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
            out = F.linear(out, outproj_weight, outproj_bias)
        else:
            assert outproj_bias is None
        ctx.save_for_backward(zxbcdt, conv1d_weight, conv1d_bias,
                              out_x, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias)
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.norm_before_gate = norm_before_gate
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        ctx.output_activation = output_activation
        ctx.threshold = threshold
        # return out if not return_final_states else (out, final_states) if not return_activation_sparsity else (out, final_states, activation_sparsity)
        if not (return_final_states or return_activation_sparsity):
            return out
        out = (out, )
        if return_final_states:
            out = out + (final_states, )
        if return_activation_sparsity:
            out = out + (activation_sparsity, )
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        zxbcdt, conv1d_weight, conv1d_bias, out, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        headdim = ctx.headdim
        nheads = D.shape[0]
        dim = nheads * headdim
        assert nheads % ctx.ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ctx.ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ctx.ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        recompute_output = outproj_weight is not None
        if recompute_output:
            out_recompute = torch.empty(*out.shape[:2], d_nonssm + dim, device=out.device, dtype=out.dtype)
            out0_recompute, out1_recompute = out_recompute.split([d_nonssm, dim], dim=-1)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        # Recompute x, B, C
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                       conv1d_weight, conv1d_bias, seq_idx, None, None, True if ctx.activation in ["silu", "swish"] else False),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)
        
        # Center C to match the centered C from forward pass
        # C = C - C.mean(dim=-1, keepdim=True)
        
        dzxbcdt = torch.empty_like(zxbcdt)
        dzx0, dz, dxBC_given, ddt_given = torch.split(dzxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
        dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)
        if outproj_weight is not None:
            dout_og = dout
            dout = F.linear(dout, outproj_weight.t())
        if d_nonssm > 0:
            dout0, dout = dout.split([d_nonssm, dim], dim=-1)
            if ctx.activation in ["silu", "swish"]:
                _swiglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
            elif ctx.activation == "relu":
                _reglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
        
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        if rmsnorm_weight is None:
            dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
            dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states, *rest = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, dz=dz, recompute_output=recompute_output, output_activation=ctx.output_activation, threshold=ctx.threshold
            )
            out_for_linear = rearrange(rest[0], "b s h p -> b s (h p)") if recompute_output else None
            drmsnorm_weight = None
        else:
            batch = dout.shape[0]
            dy_rms = rearrange(dout, "b s h p -> (b s) (h p)")
            dz = rearrange(dz, "b l d -> (b l) d")
            x_rms = rearrange(out, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            out1_recompute = rearrange(out1_recompute, "b s d -> (b s) d") if recompute_output else None
            dout, drmsnorm_weight, _, dz, *rest = _layer_norm_bwd(dy_rms, x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, None, rstd, z_rms, group_size=dim//ctx.ngroups, norm_before_gate=ctx.norm_before_gate, is_rms_norm=True, recompute_output=recompute_output, dz=dz, out=out1_recompute if recompute_output else None)
            out_for_linear = out_recompute if recompute_output else None
            dout = rearrange(dout, "(b s) (h p) -> b s h p", b=batch, p=headdim)
            dx, ddt, dA, dB, dC, dD, _, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, output_activation=ctx.output_activation, threshold=ctx.threshold
            )

        # Backpropagate through the centering operation: dC w.r.t. uncentered C
        # dC = dC - dC.mean(dim=-1, keepdim=True)
        
        # Copy gradients back into dxBC buffer
        # dxBC.copy_(torch.cat([
        #     rearrange(dx, "b l h p -> b l (h p)"),
        #     rearrange(dB, "b l g n -> b l (g n)"),
        #     rearrange(dC, "b l g n -> b l (g n)")
        # ], dim=-1))
        
        if outproj_weight is not None:
            doutproj_weight = torch.einsum("bso,bsd->od", dout_og, out_for_linear)
            doutproj_bias = dout_og.sum(dim=(0, 1)) if outproj_bias is not None else None
        else:
            doutproj_weight, doutproj_bias = None, None
        dxBC_given = rearrange(dxBC_given, "b s d -> b d s")
        dxBC_given_update, dweight, dbias, *_ = causal_conv1d_bwd_function(
            rearrange_and_update_stride(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias,
            rearrange(dxBC, "b s d -> b d s"), seq_idx, None, None, rearrange_and_update_stride(dxBC_given), False, True if ctx.activation in ["silu", "swish"] else False
        )
        if dxBC_given.stride() != dxBC_given_update.stride():
            dxBC_given.copy_(dxBC_given_update)
        else:
            dxBC_given = dxBC_given_update
        dxBC_given = rearrange(dxBC_given, "b d s -> b s d")
        return dzxbcdt, dweight, dbias, ddt_bias, dA, dD, None, dinitial_states, None, None, None, None, drmsnorm_weight, None, doutproj_weight, doutproj_bias, None, None, None, None, None, None


def mamba_split_conv1d_scan_combined(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold = None):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    return MambaSplitConv1dScanCombinedFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx, dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate, return_activation_sparsity, output_activation, threshold)


def mamba_split_conv1d_scan_ref(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, dt_limit=(0.0, float("inf")), activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    if D.dim() == 1:
        assert headdim is not None
        nheads, = D.shape
    else:
        nheads, headdim = D.shape
    assert nheads % ngroups == 0
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // ngroups // 2
    assert zxbcdt.shape == (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads)
    assert dt_bias.shape == (nheads,)
    assert A.shape == (nheads,)
    if rmsnorm_weight is not None:
        assert rmsnorm_weight.shape == (dim,)
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * ngroups * dstate, nheads], dim=-1)
    xBC = rearrange(causal_conv1d_fn(rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation=activation),
                    "b d s -> b s d")
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
    out = ssd_selective_scan(x, dt.to(x.dtype), A, B, C, D=D.float(),
                             z=z if rmsnorm_weight is None else None, dt_bias=dt_bias, dt_softplus=True, dt_limit=dt_limit)
    out = rearrange(out, "b s h p -> b s (h p)")
    if rmsnorm_weight is not None:
        out = rmsnorm_fn(out, rmsnorm_weight, None, z=rearrange(z, "b l h p -> b l (h p)"), eps=rmsnorm_eps,
                         norm_before_gate=norm_before_gate)
    if outproj_weight is not None:
        out = F.linear(out, outproj_weight, outproj_bias)
    return out


#=======================delta=======================
class MambaSplitConv1dScanCombinedDeltaFn(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
#                 rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None,
#                 ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold = 0.1):
#         # assert activation in [None, "silu", "swish", "relu"]
#         if D.dim() == 1:
#             assert headdim is not None
#             nheads, = D.shape
#         else:
#             nheads, headdim = D.shape
#         batch, seqlen, _ = zxbcdt.shape
#         dim = nheads * headdim
#         assert nheads % ngroups == 0
#         dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
#         d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
#         assert d_nonssm >= 0
#         assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
#         assert dt_bias.shape == (nheads,)
#         assert A.shape == (nheads,)
#         zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
#         seq_idx = seq_idx.contiguous() if seq_idx is not None else None
#         xBC_conv = rearrange(
#             causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
#                                                  conv1d_weight, conv1d_bias, seq_idx, None, None, True if activation in ["silu", "swish"] else None),
#             "b d s -> b s d"
#         )
#         x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
#         x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
#         B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
#         C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
#         z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
#         if rmsnorm_weight is None:
#             out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation)
#             if return_activation_sparsity:
#                 activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
#             out = rearrange(out, "b s h p -> b s (h p)")
#             rstd = None
#             if d_nonssm > 0:
#                 if activation in ["silu", "swish"]:
#                     out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
#                 else:
#                     out = torch.cat([_reglu_fwd(zx0), out], dim=-1)
#         else:
#             out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation)
#             # reshape input data into 2D tensor
            
#             # rearranged_x = rearrange(out_x, "b s h p -> b s (h p)")
#             # total_0 = (rearranged_x[-1] == 0).float().mean(-1).mean()
#             # print(total_0)
#             x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
#             z_rms = rearrange(z, "b s h p -> (b s) (h p)")
#             rmsnorm_weight = rmsnorm_weight.contiguous()
#             if d_nonssm == 0:
#                 out = None
#             else:
#                 out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
#                 out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
#                 if activation == "silu" or activation == "swish":
#                     out01[..., :d_nonssm] = _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
#                 elif activation == "relu":
#                     out01[..., :d_nonssm] = _reglu_fwd(zx0, out=out01[..., :d_nonssm])
                    
#             out, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=out,
#                                            group_size=dim // ngroups,
#                                            norm_before_gate=norm_before_gate, is_rms_norm=True)
#             if d_nonssm == 0:
#                 out = rearrange(out, "(b s) d -> b s d", b=batch)
#             else:
#                 out = out01
#         ctx.outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
#         if outproj_weight is not None:
#             if torch.is_autocast_enabled():
#                 dtype = torch.get_autocast_gpu_dtype()
#                 out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
#                 outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
#             out_prev = torch.cat([
#                     torch.zeros_like(out[:, :1, :]),  # First token has no previous
#                     out[:, :-1, :]  # Shift by 1 position
#                 ], dim=1)
                
#             diff = out - out_prev
#             diff_norm = torch.abs(diff)
#             is_changed = (diff_norm > threshold)
#             delta = torch.where(is_changed, diff, torch.zeros_like(diff))
#             if return_activation_sparsity:
#                 activation_sparsity = calculate_activation_sparsity(delta)  
#             current_out = F.linear(delta, outproj_weight, outproj_bias)
#             out = torch.cumsum(current_out, dim=1)  
#             # out = F.linear(out, outproj_weight, outproj_bias)
#             regularization_term = (diff**2).sum(dim=-1).mean()
#             # Also calculate L2 norm
#         else:
#             assert outproj_bias is None
#         ctx.save_for_backward(zxbcdt, conv1d_weight, conv1d_bias,
#                               out_x, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias)
#         ctx.dt_limit = dt_limit
#         ctx.return_final_states = return_final_states
#         ctx.activation = activation
#         ctx.rmsnorm_eps = rmsnorm_eps
#         ctx.norm_before_gate = norm_before_gate
#         ctx.chunk_size = chunk_size
#         ctx.headdim = headdim
#         ctx.ngroups = ngroups
#         ctx.output_activation = output_activation
#         #delta
#         ctx.threshold = threshold
#         # return out if not return_final_states else (out, final_states) if not return_activation_sparsity else (out, final_states, activation_sparsity)
#         if not (return_final_states or return_activation_sparsity):
#             return out
#         out = (out, )
#         if return_final_states:
#             out = out + (final_states, )
#         if return_activation_sparsity:
#             out = out + (activation_sparsity, )
#         return out + (regularization_term, )

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout, *args):
#         zxbcdt, conv1d_weight, conv1d_bias, out, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias = ctx.saved_tensors
#         dfinal_states = args[0] if ctx.return_final_states else None
#         headdim = ctx.headdim
#         nheads = D.shape[0]
#         dim = nheads * headdim
#         assert nheads % ctx.ngroups == 0
#         dstate = (conv1d_weight.shape[0] - dim) // ctx.ngroups // 2
#         d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ctx.ngroups * dstate - nheads) // 2
#         assert d_nonssm >= 0
        
#         # Split input to get zx0 and z
#         zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
#         z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        
#         # Delta encoding backward pass
#         dcurrent_out = None
#         delta = None
#         if outproj_weight is not None:
#             dout_og = dout
            
#             # Recompute the actual input to output projection (after RMSNorm/gating)
#             if rmsnorm_weight is None:
#                 # No RMSNorm case: out_x was just rearranged and possibly concatenated with non-SSM
#                 out_for_delta = rearrange(out, "b s h p -> b s (h p)")
#                 if d_nonssm > 0:
#                     # Recompute non-SSM part
#                     if ctx.activation in ["silu", "swish"]:
#                         nonssm_out = _swiglu_fwd(zx0)
#                     else:
#                         nonssm_out = _reglu_fwd(zx0)
#                     out_for_delta = torch.cat([nonssm_out, out_for_delta], dim=-1)
#             else:
#                 # RMSNorm case: need to recompute RMSNorm output
#                 x_rms = rearrange(out, "b s h p -> (b s) (h p)")
#                 z_rms = rearrange(z, "b s h p -> (b s) (h p)")
                
#                 if d_nonssm == 0:
#                     out_for_delta_temp = None
#                 else:
#                     batch = out.shape[0]
#                     seqlen = out.shape[1]
#                     out_for_delta_01 = torch.empty((batch, seqlen, d_nonssm + dim), device=out.device, dtype=out.dtype)
#                     out_for_delta_temp = rearrange(out_for_delta_01[..., d_nonssm:], "b s d -> (b s) d")
#                     if ctx.activation in ["silu", "swish"]:
#                         out_for_delta_01[..., :d_nonssm] = _swiglu_fwd(zx0, out=out_for_delta_01[..., :d_nonssm])
#                     elif ctx.activation == "relu":
#                         out_for_delta_01[..., :d_nonssm] = _reglu_fwd(zx0, out=out_for_delta_01[..., :d_nonssm])
                
#                 out_for_delta_temp, _, _ = _layer_norm_fwd(x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, z_rms, 
#                                                              out=out_for_delta_temp, group_size=dim//ctx.ngroups,
#                                                              norm_before_gate=ctx.norm_before_gate, is_rms_norm=True)
#                 if d_nonssm == 0:
#                     out_for_delta = rearrange(out_for_delta_temp, "(b s) d -> b s d", b=out.shape[0])
#                 else:
#                     out_for_delta = out_for_delta_01
            
#             # Recompute delta from the correct pre-projection tensor
#             out_prev = torch.cat([
#                 torch.zeros_like(out_for_delta[:, :1, :]),
#                 out_for_delta[:, :-1, :]
#             ], dim=1)
#             diff = out_for_delta - out_prev
#             is_changed = (torch.abs(diff) > ctx.threshold).float()
#             delta = diff * is_changed
            
#             # Backprop through cumsum: gradient of cumsum is reverse cumsum
#             dcurrent_out = torch.flip(torch.cumsum(torch.flip(dout_og, dims=[1]), dim=1), dims=[1])
            
#             # Backprop through linear projection
#             ddelta = F.linear(dcurrent_out, outproj_weight.t())
            
#             # Backprop through masking
#             ddiff = ddelta * is_changed
            
#             # Backprop through diff = out - out_prev
#             dout = ddiff.clone()
#             dout_prev = -ddiff
#             dout[:, :-1, :] += dout_prev[:, 1:, :]
        
#         # Standard backward pass setup
#         recompute_output = outproj_weight is not None
#         if recompute_output:
#             out_recompute = torch.empty(*out.shape[:2], d_nonssm + dim, device=out.device, dtype=out.dtype)
#             out0_recompute, out1_recompute = out_recompute.split([d_nonssm, dim], dim=-1)
        
#         # Recompute x, B, C
#         xBC_conv = rearrange(
#             causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
#                                        conv1d_weight, conv1d_bias, seq_idx, None, None, True if ctx.activation in ["silu", "swish"] else False),
#             "b d s -> b s d"
#         )
#         x, B, C = torch.split(xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
#         x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
#         B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
#         C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)
        
#         dzxbcdt = torch.empty_like(zxbcdt)
#         dzx0, dz, dxBC_given, ddt_given = torch.split(dzxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
#         dxBC = torch.empty_like(xBC)
#         dx, dB, dC = torch.split(dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
#         dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
#         dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
#         dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)
        
#         # Split gradient for non-SSM and SSM components
#         if d_nonssm > 0:
#             dout0, dout = dout.split([d_nonssm, dim], dim=-1)
#             if ctx.activation in ["silu", "swish"]:
#                 _swiglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
#             elif ctx.activation == "relu":
#                 _reglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
        
#         dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        
#         # Backward through RMSNorm/gating and Mamba SSM
#         if rmsnorm_weight is None:
#             dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
#             dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states, *rest = _mamba_chunk_scan_combined_bwd(
#                 dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, dz=dz, recompute_output=recompute_output, output_activation=ctx.output_activation
#             )
#             out_for_linear = rearrange(rest[0], "b s h p -> b s (h p)") if recompute_output else None
#             drmsnorm_weight = None
#         else:
#             batch = dout.shape[0]
#             dy_rms = rearrange(dout, "b s h p -> (b s) (h p)")
#             dz = rearrange(dz, "b l d -> (b l) d")
#             x_rms = rearrange(out, "b s h p -> (b s) (h p)")
#             z_rms = rearrange(z, "b s h p -> (b s) (h p)")
#             out1_recompute = rearrange(out1_recompute, "b s d -> (b s) d") if recompute_output else None
#             dout, drmsnorm_weight, _, dz, *rest = _layer_norm_bwd(dy_rms, x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, None, rstd, z_rms, group_size=dim//ctx.ngroups, norm_before_gate=ctx.norm_before_gate, is_rms_norm=True, recompute_output=recompute_output, dz=dz, out=out1_recompute if recompute_output else None)
#             out_for_linear = out_recompute if recompute_output else None
#             dout = rearrange(dout, "(b s) (h p) -> b s h p", b=batch, p=headdim)
#             dx, ddt, dA, dB, dC, dD, _, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
#                 dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, output_activation=ctx.output_activation
#             )

#         # Compute output projection weight gradients
#         if outproj_weight is not None:
#             # align dtype with gradients (FP32)
#             if delta.dtype != dcurrent_out.dtype:
#                 delta = delta.to(dcurrent_out.dtype)

#             doutproj_weight = torch.einsum("bso,bsd->od", dcurrent_out, delta)
#             doutproj_bias = dcurrent_out.sum(dim=(0, 1)) if outproj_bias is not None else None
#         else:
#             doutproj_weight, doutproj_bias = None, None

        
#         # Backward through conv1d
#         dxBC_given = rearrange(dxBC_given, "b s d -> b d s")
#         dxBC_given_update, dweight, dbias, *_ = causal_conv1d_bwd_function(
#             rearrange_and_update_stride(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias,
#             rearrange(dxBC, "b s d -> b d s"), seq_idx, None, None, rearrange_and_update_stride(dxBC_given), False, True if ctx.activation in ["silu", "swish"] else False
#         )
#         if dxBC_given.stride() != dxBC_given_update.stride():
#             dxBC_given.copy_(dxBC_given_update)
#         else:
#             dxBC_given = dxBC_given_update
#         dxBC_given = rearrange(dxBC_given, "b d s -> b s d")
        
#         return dzxbcdt, dweight, dbias, ddt_bias, dA, dD, None, dinitial_states, None, None, None, None, drmsnorm_weight, None, doutproj_weight, doutproj_bias, None, None, None, None, None, None


# @triton.jit
# def _delta_carry_kernel(
#     x_ptr, delta_ptr, mask_ptr, last_changed_ptr,
#     batch_stride_x, seq_stride_x, dim_stride_x,
#     batch_stride_d, seq_stride_d, dim_stride_d,
#     seqlen, dim,
#     threshold,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     """
#     One program per (batch, dim). BLOCK_SIZE=16 keeps the static_range unroll
#     compact (low register pressure, high occupancy). The outer tile loop is a
#     normal Triton range loop — not unrolled — so seqlen can be arbitrary.
#     """
#     batch_idx = tl.program_id(0)
#     dim_idx   = tl.program_id(1)

#     x_base = x_ptr     + batch_idx * batch_stride_x + dim_idx * dim_stride_x
#     d_base = delta_ptr + batch_idx * batch_stride_d + dim_idx * dim_stride_d
#     m_base = mask_ptr  + batch_idx * batch_stride_d + dim_idx * dim_stride_d

#     last_changed = tl.zeros((), dtype=tl.float32)
#     n_tiles = tl.cdiv(seqlen, BLOCK_SIZE)

#     for tile_idx in range(n_tiles):
#         tile_start = tile_idx * BLOCK_SIZE
#         offsets    = tile_start + tl.arange(0, BLOCK_SIZE)
#         valid      = offsets < seqlen

#         deltas_arr = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
#         masks_arr  = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

#         for i in tl.static_range(BLOCK_SIZE):
#             t       = tile_start + i
#             in_mask = t < seqlen
#             curr    = tl.load(x_base + t * seq_stride_x,
#                               mask=in_mask, other=last_changed).to(tl.float32)
#             diff    = curr - last_changed
#             changed = (tl.abs(diff) > threshold) & in_mask

#             deltas_arr = tl.where(tl.arange(0, BLOCK_SIZE) == i,
#                                   tl.where(changed, diff, 0.0), deltas_arr)
#             masks_arr  = tl.where(tl.arange(0, BLOCK_SIZE) == i,
#                                   tl.where(changed, 1.0,  0.0), masks_arr)
#             last_changed = tl.where(changed, curr, last_changed)

#         tl.store(d_base + offsets * seq_stride_d, deltas_arr, mask=valid)
#         tl.store(m_base + offsets * seq_stride_d, masks_arr,  mask=valid)

#     tl.store(last_changed_ptr + batch_idx * dim + dim_idx, last_changed)


# def _delta_encode_fwd(x: torch.Tensor, threshold: float, block_size: int = 16):
#     assert x.is_cuda and x.is_contiguous()
#     batch, seqlen, dim = x.shape

#     delta        = torch.empty_like(x)
#     mask         = torch.empty_like(x)
#     last_changed = torch.empty(batch, dim, device=x.device, dtype=torch.float32)

#     bs_x, ss_x, ds_x = x.stride()
#     bs_d, ss_d, ds_d = delta.stride()

#     _delta_carry_kernel[(batch, dim)](
#         x, delta, mask, last_changed,
#         bs_x, ss_x, ds_x,
#         bs_d, ss_d, ds_d,
#         seqlen, dim, threshold,
#         BLOCK_SIZE=block_size,
#     )

#     return delta, mask, last_changed


# def _delta_encode_bwd_kernel(
#     grad_delta_ptr, mask_ptr, grad_x_ptr,
#     batch_stride, seq_stride, dim_stride,
#     seqlen,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     batch_idx = tl.program_id(0)
#     tile_idx  = tl.program_id(1)
#     dim_idx   = tl.program_id(2)

#     tile_start = tile_idx * BLOCK_SIZE
#     offsets    = tile_start + tl.arange(0, BLOCK_SIZE)
#     load_mask  = offsets < seqlen

#     base      = batch_idx * batch_stride + dim_idx * dim_stride
#     grad      = tl.load(grad_delta_ptr + base + offsets * seq_stride, mask=load_mask, other=0.0).to(tl.float32)
#     mask_vals = tl.load(mask_ptr       + base + offsets * seq_stride, mask=load_mask, other=0.0).to(tl.float32)

#     tl.store(grad_x_ptr + base + offsets * seq_stride, grad * mask_vals, mask=load_mask)


# def _delta_encode_bwd(grad_delta: torch.Tensor, mask: torch.Tensor, block_size: int = 256):
#     grad_x  = torch.empty_like(grad_delta)
#     batch, seqlen, dim = grad_delta.shape
#     BLOCK_SIZE = block_size
#     n_tiles    = triton.cdiv(seqlen, BLOCK_SIZE)
#     bs, ss, ds = grad_delta.stride()

#     _delta_encode_bwd_kernel[(batch, n_tiles, dim)](
#         grad_delta, mask, grad_x,
#         bs, ss, ds,
#         seqlen,
#         BLOCK_SIZE=BLOCK_SIZE,
#     )
#     return grad_x


# class DeltaEncodeTriton(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, out, threshold):
#         # Ensure float32 contiguous before passing to kernels
#         x = out.float().contiguous()
#         delta, mask, _ = _delta_encode_fwd(x, threshold)

#         ctx.save_for_backward(mask)
#         ctx.threshold = threshold
#         # Return in same dtype as input to stay compatible with downstream code
#         return delta.to(out.dtype), mask.to(out.dtype)

#     @staticmethod
#     def backward(ctx, ddelta, dmask):
#         (mask,) = ctx.saved_tensors
#         grad_x = _delta_encode_bwd(
#             ddelta.float().contiguous(),
#             mask.float().contiguous(),
#         )
#         return grad_x.to(ddelta.dtype), None


# def delta_encode_triton(out, threshold):
#     return DeltaEncodeTriton.apply(out, threshold)


# def _linear_cumsum(delta, weight):
#     """cumsum(delta @ W.T, dim=1) — no bias, intentional."""
#     return torch.cumsum(F.linear(delta, weight), dim=1)


# def _linear_cumsum_bwd_input(dcurrent_out, weight):
#     """
#     Grad w.r.t. delta: F.linear(dcurrent_out, weight.t())
#     Single matmul — no chunking needed (transient alloc, not saved).
#     """
#     return F.linear(dcurrent_out, weight.t())


# # =============================================================================
# # CORRECTED MAMBA DELTA FUNCTION (ready to paste)
# # =============================================================================
# class MambaSplitConv1dScanCombinedDeltaFn(torch.autograd.Function):
#     @staticmethod
#     @custom_fwd
#     def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
#                 initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")),
#                 return_final_states=False, activation="silu",
#                 rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None,
#                 headdim=None, ngroups=1, norm_before_gate=True,
#                 return_activation_sparsity=False, output_activation=None, threshold=0.1):

#         if D.dim() == 1:
#             assert headdim is not None
#             nheads = D.shape[0]
#         else:
#             nheads, headdim = D.shape
#         batch, seqlen, _ = zxbcdt.shape
#         dim = nheads * headdim

#         assert nheads % ngroups == 0
#         dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
#         d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2

#         zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
#         seq_idx = seq_idx.contiguous() if seq_idx is not None else None

#         xBC_conv = rearrange(
#             causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
#                                        conv1d_weight, conv1d_bias, seq_idx, None, None,
#                                        True if activation in ["silu", "swish"] else None),
#             "b d s -> b s d"
#         )
#         x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
#         x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
#         B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
#         C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
#         z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None

#         # Mamba core
#         if rmsnorm_weight is None:
#             out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(
#                 x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias,
#                 initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True,
#                 dt_limit=dt_limit, output_activation=output_activation
#             )
#             if return_activation_sparsity:
#                 activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))
#             out = rearrange(out, "b s h p -> b s (h p)")
#             rstd = None
#             if d_nonssm > 0:
#                 if activation in ["silu", "swish"]:
#                     out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
#                 else:
#                     out = torch.cat([_reglu_fwd(zx0), out], dim=-1)
#             out_before_proj = out
#         else:
#             # RMSNorm branch
#             out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(
#                 x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias,
#                 initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True,
#                 dt_limit=dt_limit, output_activation=output_activation
#             )
#             x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
#             z_rms = rearrange(z, "b s h p -> (b s) (h p)")
#             rmsnorm_weight = rmsnorm_weight.contiguous()

#             if d_nonssm > 0:
#                 out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
#                 out_for_ln = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
#                 if activation in ["silu", "swish"]:
#                     out01[..., :d_nonssm] = _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
#                 elif activation == "relu":
#                     out01[..., :d_nonssm] = _reglu_fwd(zx0, out=out01[..., :d_nonssm])
#             else:
#                 out01 = None
#                 out_for_ln = None

#             out_ln, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms,
#                                               out=out_for_ln, group_size=dim // ngroups,
#                                               norm_before_gate=norm_before_gate, is_rms_norm=True)

#             if d_nonssm == 0:
#                 out = rearrange(out_ln, "(b s) d -> b s d", b=batch)
#             else:
#                 out = out01
#             out_before_proj = out[..., d_nonssm:] if d_nonssm > 0 else out

#         # DELTA ENCODING (only on SSM hidden state)
#         ctx.outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
#         regularization_term = torch.tensor(0.0, device=out.device, dtype=torch.float32)

#         if outproj_weight is not None:
#             if torch.is_autocast_enabled():
#                 dtype = torch.get_autocast_gpu_dtype()
#                 out_before_proj = out_before_proj.to(dtype)
#                 outproj_weight = outproj_weight.to(dtype)

#             delta, mask = delta_encode_triton(out_before_proj, threshold)

#             if return_activation_sparsity:
#                 activation_sparsity = (mask == 0).float().mean()

#             # Chunked linear+cumsum — avoids allocating (B, seqlen, dim_out) at once
#             out_projected = _linear_cumsum(delta, outproj_weight)

#             if d_nonssm > 0:
#                 out = torch.cat([out[..., :d_nonssm], out_projected], dim=-1)
#             else:
#                 out = out_projected

#             regularization_term = (delta ** 2).sum(dim=-1).mean()

#             ctx.save_for_backward(
#                 zxbcdt, conv1d_weight, conv1d_bias, out_x, A, D, dt_bias,
#                 initial_states, seq_idx, rmsnorm_weight, rstd,
#                 outproj_weight, outproj_bias, delta, mask, out_before_proj,
#             )
#             ctx.has_outproj = True
#         else:
#             ctx.save_for_backward(
#                 zxbcdt, conv1d_weight, conv1d_bias, out_x, A, D, dt_bias,
#                 initial_states, seq_idx, rmsnorm_weight, rstd,
#                 None, None, None, None, None,
#             )
#             ctx.has_outproj = False

#         ctx.dt_limit = dt_limit
#         ctx.return_final_states = return_final_states
#         ctx.activation = activation
#         ctx.rmsnorm_eps = rmsnorm_eps
#         ctx.norm_before_gate = norm_before_gate
#         ctx.chunk_size = chunk_size
#         ctx.headdim = headdim
#         ctx.ngroups = ngroups
#         ctx.output_activation = output_activation
#         ctx.threshold = threshold
#         ctx.d_nonssm = d_nonssm
#         ctx.dim = dim
#         ctx.dstate = dstate

#         if not (return_final_states or return_activation_sparsity):
#             return out, regularization_term

#         out_tuple = (out,)
#         if return_final_states:
#             out_tuple = out_tuple + (final_states,)
#         if return_activation_sparsity:
#             out_tuple = out_tuple + (activation_sparsity,)
#         return out_tuple + (regularization_term,)

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout, *args):
#         zxbcdt, conv1d_weight, conv1d_bias, out_x, A, D, dt_bias = ctx.saved_tensors[:7]
#         initial_states, seq_idx, rmsnorm_weight, rstd = ctx.saved_tensors[7:11]
#         outproj_weight, outproj_bias = ctx.saved_tensors[11:13]
#         delta, mask, out_before_proj = ctx.saved_tensors[13:16]

#         dfinal_states = args[0] if ctx.return_final_states else None

#         nheads = D.shape[0] if D.dim() == 1 else D.shape[0]
#         dim = ctx.dim
#         d_nonssm = ctx.d_nonssm
#         dstate = ctx.dstate

#         zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
#         z = rearrange(z, "b l (h p) -> b l h p", h=nheads)

#         if ctx.has_outproj and outproj_weight is not None:
#             if d_nonssm > 0:
#                 dout_nonssm, dout_ssm = dout.split([d_nonssm, dim], dim=-1)
#             else:
#                 dout_nonssm = None
#                 dout_ssm = dout

#             # Reverse cumsum of upstream grad (suffix-sum = grad through cumsum)
#             dcurrent_out = torch.flip(
#                 torch.cumsum(torch.flip(dout_ssm, dims=[1]), dim=1), dims=[1]
#             )

#             # Chunked matmul backward for input grad (avoids B×T×dim_in alloc at once)
#             ddelta = _linear_cumsum_bwd_input(dcurrent_out, outproj_weight)
#             dout_before_proj_ssm = ddelta * mask

#             # Weight grad output is (dim_out, dim_in) — small, no OOM risk
#             doutproj_weight = torch.einsum("bso,bsd->od", dcurrent_out, delta)
#             doutproj_bias = dcurrent_out.sum(dim=(0, 1)) if outproj_bias is not None else None

#             dout_before_proj = dout_before_proj_ssm
#         else:
#             doutproj_weight = doutproj_bias = None
#             if d_nonssm > 0:
#                 dout_nonssm, dout_before_proj = dout.split([d_nonssm, dim], dim=-1)
#             else:
#                 dout_before_proj = dout

#         if d_nonssm > 0:
#             if ctx.activation in ["silu", "swish"]:
#                 _swiglu_bwd(zx0, dout_nonssm, dxy=None, recompute_output=False)
#             else:
#                 _reglu_bwd(zx0, dout_nonssm, dxy=None, recompute_output=False)

#         xBC_conv = rearrange(
#             causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
#                                        conv1d_weight, conv1d_bias, seq_idx, None, None,
#                                        True if ctx.activation in ["silu", "swish"] else False),
#             "b d s -> b s d"
#         )
#         x, B, C = torch.split(xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
#         x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
#         B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
#         C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)

#         dzxbcdt = torch.empty_like(zxbcdt)
#         dzx0, dz, dxBC_given, ddt_given = torch.split(dzxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)

#         dxBC = torch.empty_like(xBC)
#         dx, dB, dC = torch.split(dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
#         dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
#         dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
#         dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)

#         if rmsnorm_weight is None:
#             dout_before_proj = rearrange(dout_before_proj, "b s (h p) -> b s h p", p=ctx.headdim)
#             dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
#             dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states, *_ = _mamba_chunk_scan_combined_bwd(
#                 dout_before_proj, x, dt, A, B, C, out_x, ctx.chunk_size, D=D, z=z,
#                 dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states,
#                 seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit,
#                 dx=dx, ddt=ddt_given, dB=dB, dC=dC, dz=dz, recompute_output=False,
#                 output_activation=ctx.output_activation
#             )
#             drmsnorm_weight = None
#         else:
#             batch = dout_before_proj.shape[0]
#             dy_rms = rearrange(dout_before_proj, "b s (h p) -> (b s) (h p)", p=ctx.headdim)
#             dz = rearrange(dz, "b l d -> (b l) d")
#             x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
#             z_rms = rearrange(z, "b s h p -> (b s) (h p)")

#             dout_rms, drmsnorm_weight, _, dz, *_ = _layer_norm_bwd(
#                 dy_rms, x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, None, rstd, z_rms,
#                 group_size=dim // ctx.ngroups, norm_before_gate=ctx.norm_before_gate,
#                 is_rms_norm=True, recompute_output=False, dz=dz
#             )
#             dout_rms = rearrange(dout_rms, "(b s) (h p) -> b s h p", b=batch, p=ctx.headdim)

#             dx, ddt, dA, dB, dC, dD, _, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
#                 dout_rms, x, dt, A, B, C, out_x, ctx.chunk_size, D=D, z=None,
#                 dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states,
#                 seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit,
#                 dx=dx, ddt=ddt_given, dB=dB, dC=dC, output_activation=ctx.output_activation
#             )

#         dxBC_given = rearrange(dxBC_given, "b s d -> b d s")
#         dxBC_given_update, dweight, dbias, *_ = causal_conv1d_bwd_function(
#             rearrange_and_update_stride(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias,
#             rearrange(dxBC, "b s d -> b d s"), seq_idx, None, None,
#             rearrange_and_update_stride(dxBC_given), False,
#             True if ctx.activation in ["silu", "swish"] else False
#         )
#         if dxBC_given.stride() != dxBC_given_update.stride():
#             dxBC_given.copy_(dxBC_given_update)
#         else:
#             dxBC_given = dxBC_given_update
#         dxBC_given = rearrange(dxBC_given, "b d s -> b s d")

#         return (dzxbcdt, dweight, dbias, ddt_bias, dA, dD, None, dinitial_states, None, None, None, None,
#                 drmsnorm_weight, None, doutproj_weight, doutproj_bias, None, None, None, None, None, None)


def mamba_split_conv1d_scan_combined_Delta(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold = None):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    return MambaSplitConv1dScanCombinedDeltaFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx, dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate, return_activation_sparsity, output_activation, threshold)


#==========================================dr====================================
class DR_SSM_SplitConv1dScanCombinedFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, D, chunk_size, initial_states=None, seq_idx=None, return_final_states=False, activation="silu",
                rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None,
                ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold=0.0):
        # assert activation in [None, "silu", "swish", "relu"]
        if D.dim() == 1:
            assert headdim is not None
            nheads, = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        assert nheads % ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 5 * dim - 2 * ngroups * dstate) // 2
        assert d_nonssm >= 0
        assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 5 * dim + 2 * ngroups * dstate)
        print('we are here hhhhhhhhhhhhhhh')
        zx0, z, E, I, A, xBC = torch.split(zxbcdt, [2 * d_nonssm, dim,dim,dim,dim, dim + ngroups * dstate * 2], dim=-1)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                                 conv1d_weight, conv1d_bias, seq_idx, None, None, True if activation in ["silu", "swish"] else None),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        E = rearrange(E, "b l (h p) -> b l h p", h=nheads)
        I = rearrange(I, "b l (h p) -> b l h p", h=nheads)
        A = rearrange(A, "b l (h p) -> b l h p", h=nheads)
        A = -F.softplus(A)
        E = F.relu(E)
        I = F.softplus(I)
        
        if rmsnorm_weight is None:
            out, out_x, sigma_squared_sum, dA_cumsum, states, final_states, x_masked, M, sigmoid_delta, E_mean = _dr_ssm_chunk_scan_combined_fwd(x, A, B, C, E, I, chunk_size=chunk_size, D=D, z=z, initial_states=initial_states, seq_idx=seq_idx,  output_activation=output_activation, threshold=threshold)
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            out = rearrange(out, "b s h p -> b s (h p)")
            rstd = None
            if d_nonssm > 0:
                if activation in ["silu", "swish"]:
                    out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
                else:
                    out = torch.cat([_reglu_fwd(zx0), out], dim=-1)
        else:
            out, out_x, sigma_squared_sum, dA_cumsum, states, final_states, x_masked, M, sigmoid_delta, E_mean = _dr_ssm_chunk_scan_combined_fwd(x, A, B, C, E, I, chunk_size=chunk_size, D=D, z=None, initial_states=initial_states, seq_idx=seq_idx, output_activation=output_activation, threshold=threshold)
            # When z=None, out_x is None, so use out instead for the computations
            out_x = out  # Use out as out_x since z=None means no gating
            # reshape input data into 2D tensor
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            # rearranged_x = rearrange(out_x, "b s h p -> b s (h p)")
            # total_0 = (rearranged_x[-1] == 0).float().mean(-1).mean()
            # print(total_0)
            x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            rmsnorm_weight = rmsnorm_weight.contiguous()
            if d_nonssm == 0:
                out = None
            else:
                out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
                out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
                if activation == "silu" or activation == "swish":
                    out01[..., :d_nonssm] = _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
                elif activation == "relu":
                    out01[..., :d_nonssm] = _reglu_fwd(zx0, out=out01[..., :d_nonssm])
                    
            out, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=out,
                                           group_size=dim // ngroups,
                                           norm_before_gate=norm_before_gate, is_rms_norm=True)
            if d_nonssm == 0:
                out = rearrange(out, "(b s) d -> b s d", b=batch)
            else:
                out = out01
        ctx.outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
        
        # apply delta rule on y
        if outproj_weight is not None:
            # Force dtype alignment between activations and weights regardless of autocast
            target_dtype = outproj_weight.dtype
            if out.dtype != target_dtype:
                out = out.to(target_dtype)
            
            out_prev = torch.cat([
                torch.zeros_like(out[:, :1, :]),  # First token has no previous
                out[:, :-1, :]  # Shift by 1 position
            ], dim=1)
            
            diff = out - out_prev
            diff_norm = torch.abs(diff)
            is_changed = (diff_norm > threshold)
            delta = torch.where(is_changed, diff, torch.zeros_like(diff))             
            # Ensure bias matches weight dtype
            bias = outproj_bias.to(target_dtype) if outproj_bias is not None else None
            
            current_out = F.linear(delta, outproj_weight, bias)
            out = torch.cumsum(current_out, dim=1)
            # out = F.linear(out, outproj_weight, outproj_bias)
        else:
            assert outproj_bias is None
        ctx.save_for_backward(zxbcdt, conv1d_weight, conv1d_bias,
                              out_x, D, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias)
        # ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.norm_before_gate = norm_before_gate
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        ctx.output_activation = output_activation
        ctx.threshold = threshold
        # return out if not return_final_states else (out, final_states) if not return_activation_sparsity else (out, final_states, activation_sparsity)
        if not (return_final_states or return_activation_sparsity):
            return out
        # Reduce sigma_squared_sum to scalar for regularization loss
        regularization_scalar = sigma_squared_sum.mean()
        out = (out, regularization_scalar)
        if return_final_states:
            out = out + (final_states, )
        if return_activation_sparsity:
            out = out + (activation_sparsity, )
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        zxbcdt, conv1d_weight, conv1d_bias, out, D, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        headdim = ctx.headdim
        nheads = D.shape[0]
        dim = nheads * headdim
        dstate = (conv1d_weight.shape[0] - dim) // ctx.ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 5 * dim - 2 * ctx.ngroups * dstate) // 2
        assert nheads % ctx.ngroups == 0
        try:
            assert zxbcdt.shape == (dout.shape[0], dout.shape[1], 2 * d_nonssm + 5 * dim + 2 * ctx.ngroups * dstate)
        except AssertionError as e:
            print("Shape mismatch in backward:", zxbcdt.shape)
        
        assert d_nonssm >= 0
        recompute_output = outproj_weight is not None
        if recompute_output:
            out_recompute = torch.empty(*out.shape[:2], d_nonssm + dim, device=out.device, dtype=out.dtype)
            out0_recompute, out1_recompute = out_recompute.split([d_nonssm, dim], dim=-1)
        zx0, z, E, I, A, xBC = torch.split(zxbcdt, [2 * d_nonssm, dim,dim, dim, dim, dim + 2 * ctx.ngroups * dstate], dim=-1)
        # Recompute x, B, C
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                       conv1d_weight, conv1d_bias, seq_idx, None, None, True if ctx.activation in ["silu", "swish"] else False),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)
        E = rearrange(E, "b l (h p) -> b l h p", h=nheads)
        I = rearrange(I, "b l (h p) -> b l h p", h=nheads)
        A = rearrange(A, "b l (h p) -> b l h p", h=nheads)
        A = -F.softplus(A)
        E = F.relu(E)
        I = F.softplus(I)
        
        dzxbcdt = torch.empty_like(zxbcdt)
        dzx0, dz,dE_given, dI_given, dA_given, dxBC_given = torch.split(dzxbcdt, [2 * d_nonssm, dim,dim, dim, dim, dim + 2 * ctx.ngroups * dstate], dim=-1)
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
        dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)
        dE = rearrange(dE_given, "b l (h p) -> b l h p", h=nheads)
        dI = rearrange(dI_given, "b l (h p) -> b l h p", h=nheads)
        dA = rearrange(dA_given, "b l (h p) -> b l h p", h=nheads)
        
        
        dcurrent_out = None
        delta = None
        if outproj_weight is not None:
            dout_og = dout
            
            # Recompute the actual input to output projection (after RMSNorm/gating)
            if rmsnorm_weight is None:
                # No RMSNorm case: out_x was just rearranged and possibly concatenated with non-SSM
                out_for_delta = rearrange(out, "b s h p -> b s (h p)")
                if d_nonssm > 0:
                    # Recompute non-SSM part
                    if ctx.activation in ["silu", "swish"]:
                        nonssm_out = _swiglu_fwd(zx0)
                    else:
                        nonssm_out = _reglu_fwd(zx0)
                    out_for_delta = torch.cat([nonssm_out, out_for_delta], dim=-1)
            else:
                # RMSNorm case: need to recompute RMSNorm output
                x_rms = rearrange(out, "b s h p -> (b s) (h p)")
                z_rms = rearrange(z, "b s h p -> (b s) (h p)")
                
                if d_nonssm == 0:
                    out_for_delta_temp = None
                else:
                    batch = out.shape[0]
                    seqlen = out.shape[1]
                    out_for_delta_01 = torch.empty((batch, seqlen, d_nonssm + dim), device=out.device, dtype=out.dtype)
                    out_for_delta_temp = rearrange(out_for_delta_01[..., d_nonssm:], "b s d -> (b s) d")
                    if ctx.activation in ["silu", "swish"]:
                        out_for_delta_01[..., :d_nonssm] = _swiglu_fwd(zx0, out=out_for_delta_01[..., :d_nonssm])
                    elif ctx.activation == "relu":
                        out_for_delta_01[..., :d_nonssm] = _reglu_fwd(zx0, out=out_for_delta_01[..., :d_nonssm])
                
                out_for_delta_temp, _, _ = _layer_norm_fwd(x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, z_rms, 
                                                             out=out_for_delta_temp, group_size=dim//ctx.ngroups,
                                                             norm_before_gate=ctx.norm_before_gate, is_rms_norm=True)
                if d_nonssm == 0:
                    out_for_delta = rearrange(out_for_delta_temp, "(b s) d -> b s d", b=out.shape[0])
                else:
                    out_for_delta = out_for_delta_01
            
            # Recompute delta from the correct pre-projection tensor
            out_prev = torch.cat([
                torch.zeros_like(out_for_delta[:, :1, :]),
                out_for_delta[:, :-1, :]
            ], dim=1)
            diff = out_for_delta - out_prev
            is_changed = (torch.abs(diff) > ctx.threshold).float()
            delta = diff * is_changed
            
            # Backprop through cumsum: gradient of cumsum is reverse cumsum
            dcurrent_out = torch.flip(torch.cumsum(torch.flip(dout_og, dims=[1]), dim=1), dims=[1])
            
            # Backprop through linear projection
            ddelta = F.linear(dcurrent_out, outproj_weight.t())
            
            # Backprop through masking
            ddiff = ddelta * is_changed
            
            # Backprop through diff = out - out_prev
            dout = ddiff.clone()
            dout_prev = -ddiff
            dout[:, :-1, :] += dout_prev[:, 1:, :]
        
        
        if d_nonssm > 0:
            dout0, dout = dout.split([d_nonssm, dim], dim=-1)
            if ctx.activation in ["silu", "swish"]:
                _swiglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
            elif ctx.activation == "relu":
                _reglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
        
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        if rmsnorm_weight is None:
            dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
            dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states, *rest = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, dz=dz, recompute_output=recompute_output, output_activation=ctx.output_activation, threshold=ctx.threshold
            )
            out_for_linear = rearrange(rest[0], "b s h p -> b s (h p)") if recompute_output else None
            drmsnorm_weight = None
        else:
            batch = dout.shape[0]
            dy_rms = rearrange(dout, "b s h p -> (b s) (h p)")
            dz = rearrange(dz, "b l d -> (b l) d")
            x_rms = rearrange(out, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            out1_recompute = rearrange(out1_recompute, "b s d -> (b s) d") if recompute_output else None
            # backward through rmsnorm
            dout, drmsnorm_weight, _, dz, *rest = _layer_norm_bwd(dy_rms, x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, None, rstd, z_rms, group_size=dim//ctx.ngroups, norm_before_gate=ctx.norm_before_gate, is_rms_norm=True, recompute_output=recompute_output, dz=dz, out=out1_recompute if recompute_output else None)
            out_for_linear = out_recompute if recompute_output else None
            dout = rearrange(dout, "(b s) (h p) -> b s h p", b=batch, p=headdim)
            dx, dA, dB, dC, dD, _, dE,dI, dinitial_states = _dr_ssm_chunk_scan_combined_bwd(
                dout, x, A, B, C,E, I, out, ctx.chunk_size, D=D, z=None, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx,  dx=dx, dB=dB, dC=dC,dE = dE, dI = dI, dA = dA, output_activation=ctx.output_activation, threshold=ctx.threshold
            )
            

        
        if outproj_weight is not None:
            doutproj_weight = torch.einsum("bso,bsd->od", dcurrent_out, delta.to(dcurrent_out.dtype))
            doutproj_bias = dcurrent_out.sum(dim=(0, 1)) if outproj_bias is not None else None
        else:
            doutproj_weight, doutproj_bias = None, None
            
        dxBC_given = rearrange(dxBC_given, "b s d -> b d s")
        dxBC_given_update, dweight, dbias, *_ = causal_conv1d_bwd_function(
            rearrange_and_update_stride(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias,
            rearrange(dxBC, "b s d -> b d s"), seq_idx, None, None, rearrange_and_update_stride(dxBC_given), False, True if ctx.activation in ["silu", "swish"] else False
        )
        if dxBC_given.stride() != dxBC_given_update.stride():
            dxBC_given.copy_(dxBC_given_update)
        else:
            dxBC_given = dxBC_given_update
        dxBC_given = rearrange(dxBC_given, "b d s -> b s d")
        
        dI = dI * torch.sigmoid(I)
        dE = dE * (E > 0).float()
        dA = - dA * torch.sigmoid(-A)
        
        return dzxbcdt, dweight, dbias, dD, None, dinitial_states, None, None, None, drmsnorm_weight, None, doutproj_weight, doutproj_bias, None, None, None, None, None, None


def DR_SSM_split_conv1d_scan_combined(zxbcdt, conv1d_weight, conv1d_bias, dt_bias,  chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold = None):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    return DR_SSM_SplitConv1dScanCombinedFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx,  return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate, return_activation_sparsity, output_activation, threshold)





# =============== Sub Mamba 2==================

class SubMambaSplitConv1dScanCombinedFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
                rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None,
                ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold=0.0):
        # assert activation in [None, "silu", "swish", "relu"]
        if D.dim() == 1:
            assert headdim is not None
            nheads, = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        assert nheads % ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 3 * dim - 2 * ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 3 * dim + 2 * ngroups * dstate + nheads)
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        zx0, z, I, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim,dim, dim + ngroups * dstate * 2, nheads], dim=-1)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                                 conv1d_weight, conv1d_bias, seq_idx, None, None, True if activation in ["silu", "swish"] else None),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
        I = rearrange(I, "b l (h p) -> b l h p", h=nheads)
        x_old = x
        x = _inhibition_gate_fwd(x, I, x)

        # Modify the I
        # I = F.softplus(I)
        # x_old = x
        # x = F.relu(x - I - F.softplus(x.mean(dim=-1, keepdim=True)))
        
        
        
        
        
        # C centered
        # C = C - C.mean(dim=-1, keepdim=True)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        if rmsnorm_weight is None:
            out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation, threshold=threshold)
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            out = rearrange(out, "b s h p -> b s (h p)")
            rstd = None
            if d_nonssm > 0:
                if activation in ["silu", "swish"]:
                    out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
                else:
                    out = torch.cat([_reglu_fwd(zx0), out], dim=-1)
        else:
            out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation, threshold=threshold)
            # reshape input data into 2D tensor
            out_x_old = out_x
            out_x = _inhibition_gate_fwd(x_old, I, out_x)
            
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            # rearranged_x = rearrange(out_x, "b s h p -> b s (h p)")
            # total_0 = (rearranged_x[-1] == 0).float().mean(-1).mean()
            # print(total_0)
            x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            rmsnorm_weight = rmsnorm_weight.contiguous()
            if d_nonssm == 0:
                out = None
            else:
                out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
                out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
                if activation == "silu" or activation == "swish":
                    out01[..., :d_nonssm] = _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
                elif activation == "relu":
                    out01[..., :d_nonssm] = _reglu_fwd(zx0, out=out01[..., :d_nonssm])
                    
            out, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=out,
                                           group_size=dim // ngroups,
                                           norm_before_gate=norm_before_gate, is_rms_norm=True)
            if d_nonssm == 0:
                out = rearrange(out, "(b s) d -> b s d", b=batch)
            else:
                out = out01
        ctx.outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
        if outproj_weight is not None:
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
                outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
            out = F.linear(out, outproj_weight, outproj_bias)
        else:
            assert outproj_bias is None
            
        # x = x_old
        ctx.save_for_backward(zxbcdt, conv1d_weight, conv1d_bias,
                              out_x_old, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias)
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.norm_before_gate = norm_before_gate
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        ctx.output_activation = output_activation
        ctx.threshold = threshold
        # return out if not return_final_states else (out, final_states) if not return_activation_sparsity else (out, final_states, activation_sparsity)
        if not (return_final_states or return_activation_sparsity):
            return out
        out = (out, )
        if return_final_states:
            out = out + (final_states, )
        if return_activation_sparsity:
            out = out + (activation_sparsity, )
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        zxbcdt, conv1d_weight, conv1d_bias, out, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        headdim = ctx.headdim
        nheads = D.shape[0]
        dim = nheads * headdim
        assert nheads % ctx.ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ctx.ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 3 * dim - 2 * ctx.ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        recompute_output = outproj_weight is not None
        if recompute_output:
            out_recompute = torch.empty(*out.shape[:2], d_nonssm + dim, device=out.device, dtype=out.dtype)
            out0_recompute, out1_recompute = out_recompute.split([d_nonssm, dim], dim=-1)
        zx0, z, I, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim,dim,  dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        # Recompute x, B, C
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                       conv1d_weight, conv1d_bias, seq_idx, None, None, True if ctx.activation in ["silu", "swish"] else False),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)
        I = rearrange(I, "b l (h p) -> b l h p", h=nheads)
        x_pre_gate = x  # save pre-gate x for input gate backward
        x = _inhibition_gate_fwd(x, I, x)

        dzxbcdt = torch.empty_like(zxbcdt)
        dzx0, dz, dI, dxBC_given, ddt_given = torch.split(dzxbcdt, [2 * d_nonssm, dim, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        # dx_out = torch.empty_like(dx)
        # dx_in = torch.empty_like(dx)
        dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
        dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)
        if outproj_weight is not None:
            dout_og = dout
            dout = F.linear(dout, outproj_weight.t())
        if d_nonssm > 0:
            dout0, dout = dout.split([d_nonssm, dim], dim=-1)
            if ctx.activation in ["silu", "swish"]:
                _swiglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
            elif ctx.activation == "relu":
                _reglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
        
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        if rmsnorm_weight is None:
            dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
            dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states, *rest = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, dz=dz, recompute_output=recompute_output, output_activation=ctx.output_activation, threshold=ctx.threshold
            )
            out_for_linear = rearrange(rest[0], "b s h p -> b s (h p)") if recompute_output else None
            drmsnorm_weight = None
            # Input gate backward
            dx_temp, di_in, _ = _inhibition_gate_bwd(x_pre_gate, I, x_pre_gate, dx)
            dx.copy_(dx_temp)
            dI.copy_(rearrange(di_in, "b l h p -> b l (h p)"))
        else:
            batch = dout.shape[0]
            dy_rms = rearrange(dout, "b s h p -> (b s) (h p)")
            dz = rearrange(dz, "b l d -> (b l) d")
            # Recompute the gated output for _layer_norm_bwd
            out_gated = _inhibition_gate_fwd(x_pre_gate, I, out)
            x_rms = rearrange(out_gated, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            out1_recompute = rearrange(out1_recompute, "b s d -> (b s) d") if recompute_output else None
            d_gated_out, drmsnorm_weight, _, dz, *rest = _layer_norm_bwd(dy_rms, x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, None, rstd, z_rms, group_size=dim//ctx.ngroups, norm_before_gate=ctx.norm_before_gate, is_rms_norm=True, recompute_output=recompute_output, dz=dz, out=out1_recompute if recompute_output else None)
            d_gated_out = rearrange(d_gated_out, "(b s) (h p) -> b s h p", b=batch, p=headdim)
            out_for_linear = out_recompute if recompute_output else None

            # Output gate backward
            d_pre_x_output, di_out, d_mamba_out = _inhibition_gate_bwd(x_pre_gate, I, out, d_gated_out)
            
            # Mamba backward
            dx_post, ddt, dA, dB, dC, dD, _, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
                d_mamba_out, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, output_activation=ctx.output_activation, threshold=ctx.threshold
            )
            
            # Input gate backward
            d_pre_x_input_mask, di_in, d_pre_x_input_mul = _inhibition_gate_bwd(x_pre_gate, I, x_pre_gate, dx_post)
            
            # Accumulate total gradient w.r.t. pre-gate x
            dx.copy_(d_pre_x_input_mask)
            dx.add_(d_pre_x_input_mul)
            dx.add_(d_pre_x_output)
            
            dI.copy_(rearrange((di_out + di_in), "b l h p -> b l (h p)"))

        # Backpropagate through the centering operation: dC w.r.t. uncentered C
        # dC = dC - dC.mean(dim=-1, keepdim=True)
        
        # Copy gradients back into dxBC buffer
        # dxBC.copy_(torch.cat([
        #     rearrange(dx, "b l h p -> b l (h p)"),
        #     rearrange(dB, "b l g n -> b l (g n)"),
        #     rearrange(dC, "b l g n -> b l (g n)")
        # ], dim=-1))
        
        x = x_pre_gate
        if outproj_weight is not None:
            doutproj_weight = torch.einsum("bso,bsd->od", dout_og, out_for_linear)
            doutproj_bias = dout_og.sum(dim=(0, 1)) if outproj_bias is not None else None
        else:
            doutproj_weight, doutproj_bias = None, None
        dxBC_given = rearrange(dxBC_given, "b s d -> b d s")
        dxBC_given_update, dweight, dbias, *_ = causal_conv1d_bwd_function(
            rearrange_and_update_stride(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias,
            rearrange(dxBC, "b s d -> b d s"), seq_idx, None, None, rearrange_and_update_stride(dxBC_given), False, True if ctx.activation in ["silu", "swish"] else False
        )
        if dxBC_given.stride() != dxBC_given_update.stride():
            dxBC_given.copy_(dxBC_given_update)
        else:
            dxBC_given = dxBC_given_update
        dxBC_given = rearrange(dxBC_given, "b d s -> b s d")
        return dzxbcdt, dweight, dbias, ddt_bias, dA, dD, None, dinitial_states, None, None, None, None, drmsnorm_weight, None, doutproj_weight, doutproj_bias, None, None, None, None, None, None


def sub_mamba_split_conv1d_scan_combined(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold = None):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    return SubMambaSplitConv1dScanCombinedFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx, dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate, return_activation_sparsity, output_activation, threshold)



#================== hard mamba2======================


class HardMambaSplitConv1dScanCombinedFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
                rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None,
                ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold=0.0, S = None):
        # assert activation in [None, "silu", "swish", "relu"]
        if D.dim() == 1:
            assert headdim is not None
            nheads, = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        assert nheads % ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                                 conv1d_weight, conv1d_bias, seq_idx, None, None, True if activation in ["silu", "swish"] else None),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
        S = rearrange(S, "b l (h p) -> b l h p", h=nheads) if S is not None else None
        # C centered
        # C = C - C.mean(dim=-1, keepdim=True)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        if rmsnorm_weight is None:
            out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation, threshold=threshold)
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            out = rearrange(out, "b s h p -> b s (h p)")
            rstd = None
            if d_nonssm > 0:
                if activation in ["silu", "swish"]:
                    out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
                else:
                    out = torch.cat([_reglu_fwd(zx0), out], dim=-1)
        else:
            out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation, threshold=threshold)
            # reshape input data into 2D tensor
            # out_x_old = out_x
            out_x = S * out_x
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            # rearranged_x = rearrange(out_x, "b s h p -> b s (h p)")
            # total_0 = (rearranged_x[-1] == 0).float().mean(-1).mean()
            # print(total_0)
            x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            rmsnorm_weight = rmsnorm_weight.contiguous()
            if d_nonssm == 0:
                out = None
            else:
                out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
                out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
                if activation == "silu" or activation == "swish":
                    out01[..., :d_nonssm] = _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
                elif activation == "relu":
                    out01[..., :d_nonssm] = _reglu_fwd(zx0, out=out01[..., :d_nonssm])
                    
            out, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=out,
                                           group_size=dim // ngroups,
                                           norm_before_gate=norm_before_gate, is_rms_norm=True)
            if d_nonssm == 0:
                out = rearrange(out, "(b s) d -> b s d", b=batch)
            else:
                out = out01
        ctx.outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
        if outproj_weight is not None:
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
                outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
            out = F.linear(out, outproj_weight, outproj_bias)
        else:
            assert outproj_bias is None
            
        S = rearrange(S, "b l h p -> b l (h p)", h=nheads) if S is not None else None

        ctx.save_for_backward(zxbcdt, conv1d_weight, conv1d_bias,
                              out_x, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias, S)
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.norm_before_gate = norm_before_gate
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        ctx.output_activation = output_activation
        ctx.threshold = threshold
        # return out if not return_final_states else (out, final_states) if not return_activation_sparsity else (out, final_states, activation_sparsity)
        if not (return_final_states or return_activation_sparsity):
            return out
        out = (out, )
        if return_final_states:
            out = out + (final_states, )
        if return_activation_sparsity:
            out = out + (activation_sparsity, )
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        zxbcdt, conv1d_weight, conv1d_bias, out, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias, S = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        headdim = ctx.headdim
        nheads = D.shape[0]
        dim = nheads * headdim
        assert nheads % ctx.ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ctx.ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ctx.ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        recompute_output = outproj_weight is not None
        if recompute_output:
            out_recompute = torch.empty(*out.shape[:2], d_nonssm + dim, device=out.device, dtype=out.dtype)
            out0_recompute, out1_recompute = out_recompute.split([d_nonssm, dim], dim=-1)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        # Recompute x, B, C
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                       conv1d_weight, conv1d_bias, seq_idx, None, None, True if ctx.activation in ["silu", "swish"] else False),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)
        
        # Center C to match the centered C from forward pass
        # C = C - C.mean(dim=-1, keepdim=True)
        
        dzxbcdt = torch.empty_like(zxbcdt)
        dzx0, dz, dxBC_given, ddt_given = torch.split(dzxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
        dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)
        dS = torch.empty_like(S) if S is not None else None
        dS = rearrange(dS, "b l (h p) -> b l h p", h=nheads) if S is not None else None
        
        if outproj_weight is not None:
            dout_og = dout
            dout = F.linear(dout, outproj_weight.t())
        if d_nonssm > 0:
            dout0, dout = dout.split([d_nonssm, dim], dim=-1)
            if ctx.activation in ["silu", "swish"]:
                _swiglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
            elif ctx.activation == "relu":
                _reglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
        
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        if rmsnorm_weight is None:
            dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
            dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states, *rest = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, dz=dz, recompute_output=recompute_output, output_activation=ctx.output_activation, threshold=ctx.threshold
            )
            out_for_linear = rearrange(rest[0], "b s h p -> b s (h p)") if recompute_output else None
            drmsnorm_weight = None
        else:
            batch = dout.shape[0]
            dy_rms = rearrange(dout, "b s h p -> (b s) (h p)")
            dz = rearrange(dz, "b l d -> (b l) d")
            x_rms = rearrange(out, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            out1_recompute = rearrange(out1_recompute, "b s d -> (b s) d") if recompute_output else None
            dout, drmsnorm_weight, _, dz, *rest = _layer_norm_bwd(dy_rms, x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, None, rstd, z_rms, group_size=dim//ctx.ngroups, norm_before_gate=ctx.norm_before_gate, is_rms_norm=True, recompute_output=recompute_output, dz=dz, out=out1_recompute if recompute_output else None)

            out_for_linear = out_recompute if recompute_output else None
            dout = rearrange(dout, "(b s) (h p) -> b s h p", b=batch, p=headdim)
            # out_masked = S * out if S is not None else out
            S = rearrange(S, "b l (h p) -> b l h p", h=nheads) if S is not None else None
            dS = dout * out if S is not None else None
            dout = dout * S if S is not None else dout
            dx, ddt, dA, dB, dC, dD, _, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, output_activation=ctx.output_activation, threshold=ctx.threshold
            )

        # Backpropagate through the centering operation: dC w.r.t. uncentered C
        # dC = dC - dC.mean(dim=-1, keepdim=True)
        
        # Copy gradients back into dxBC buffer
        # dxBC.copy_(torch.cat([
        #     rearrange(dx, "b l h p -> b l (h p)"),
        #     rearrange(dB, "b l g n -> b l (g n)"),
        #     rearrange(dC, "b l g n -> b l (g n)")
        # ], dim=-1))
        
        if outproj_weight is not None:
            doutproj_weight = torch.einsum("bso,bsd->od", dout_og, out_for_linear)
            doutproj_bias = dout_og.sum(dim=(0, 1)) if outproj_bias is not None else None
        else:
            doutproj_weight, doutproj_bias = None, None
        dxBC_given = rearrange(dxBC_given, "b s d -> b d s")
        dxBC_given_update, dweight, dbias, *_ = causal_conv1d_bwd_function(
            rearrange_and_update_stride(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias,
            rearrange(dxBC, "b s d -> b d s"), seq_idx, None, None, rearrange_and_update_stride(dxBC_given), False, True if ctx.activation in ["silu", "swish"] else False
        )
        if dxBC_given.stride() != dxBC_given_update.stride():
            dxBC_given.copy_(dxBC_given_update)
        else:
            dxBC_given = dxBC_given_update
        dxBC_given = rearrange(dxBC_given, "b d s -> b s d")
        dS = rearrange(dS, "b l h p -> b l (h p)") if dS is not None else None
        # dinitial_states = rearrange(dinitial_states, "b l h p -> b l (h p)")
        return dzxbcdt, dweight, dbias, ddt_bias, dA, dD, None, dinitial_states, None, None, None, None, drmsnorm_weight, None, doutproj_weight, doutproj_bias, None, None, None, None, None, None, dS


def hard_mamba_split_conv1d_scan_combined(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold = None, S = None):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    return HardMambaSplitConv1dScanCombinedFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx, dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate, return_activation_sparsity, output_activation, threshold, S)



# =================== sfa=====================

class SfaMambaSplitConv1dScanCombinedFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
                rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None,
                ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold=0.0):
        # assert activation in [None, "silu", "swish", "relu"]
        if D.dim() == 1:
            assert headdim is not None
            nheads, = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        assert nheads % ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                                 conv1d_weight, conv1d_bias, seq_idx, None, None, True if activation in ["silu", "swish"] else None),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
        
        # C centered
        # C = C - C.mean(dim=-1, keepdim=True)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        if rmsnorm_weight is None:
            out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation, threshold=threshold)
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            out = rearrange(out, "b s h p -> b s (h p)")
            rstd = None
            if d_nonssm > 0:
                if activation in ["silu", "swish"]:
                    out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
                else:
                    out = torch.cat([_reglu_fwd(zx0), out], dim=-1)
        else:
            out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, output_activation=output_activation, threshold=threshold)
            # reshape input data into 2D tensor
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(rearrange(out_x, "b s h p -> b s (h p)"))  
            # rearranged_x = rearrange(out_x, "b s h p -> b s (h p)")
            # total_0 = (rearranged_x[-1] == 0).float().mean(-1).mean()
            # print(total_0)
            x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            rmsnorm_weight = rmsnorm_weight.contiguous()
            if d_nonssm == 0:
                out = None
            else:
                out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
                out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
                if activation == "silu" or activation == "swish":
                    out01[..., :d_nonssm] = _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
                elif activation == "relu":
                    out01[..., :d_nonssm] = _reglu_fwd(zx0, out=out01[..., :d_nonssm])
                    
            out, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=out,
                                           group_size=dim // ngroups,
                                           norm_before_gate=norm_before_gate, is_rms_norm=True)
            if d_nonssm == 0:
                out = rearrange(out, "(b s) d -> b s d", b=batch)
            else:
                out = out01
        ctx.outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
        if outproj_weight is not None:
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
                outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
            out_prev = torch.cat([
                    torch.zeros_like(out[:, :1, :]),  # First token has no previous
                    out[:, :-1, :]  # Shift by 1 position
                ], dim=1)
                
            diff = out - out_prev
            diff_norm = torch.abs(diff)
            is_changed = (diff_norm > threshold).float()
            delta = diff * is_changed.expand_as(diff)
            if return_activation_sparsity:
                activation_sparsity = calculate_activation_sparsity(delta)  
            current_out = F.linear(delta, outproj_weight, outproj_bias)
            out = torch.cumsum(current_out, dim=1)  
            # out = F.linear(out, outproj_weight, outproj_bias)
            # regularization_term = (diff**2).sum()
            # Also calculate L2 norm
        else:
            assert outproj_bias is None
        ctx.save_for_backward(zxbcdt, conv1d_weight, conv1d_bias,
                              out_x, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias)
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.norm_before_gate = norm_before_gate
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        ctx.output_activation = output_activation
        ctx.threshold = threshold
        # return out if not return_final_states else (out, final_states) if not return_activation_sparsity else (out, final_states, activation_sparsity)
        if not (return_final_states or return_activation_sparsity):
            return out
        out = (out, )
        if return_final_states:
            out = out + (final_states, )
        if return_activation_sparsity:
            out = out + (activation_sparsity, )
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        zxbcdt, conv1d_weight, conv1d_bias, out, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        headdim = ctx.headdim
        nheads = D.shape[0]
        dim = nheads * headdim
        assert nheads % ctx.ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ctx.ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ctx.ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        recompute_output = outproj_weight is not None
        if recompute_output:
            out_recompute = torch.empty(*out.shape[:2], d_nonssm + dim, device=out.device, dtype=out.dtype)
            out0_recompute, out1_recompute = out_recompute.split([d_nonssm, dim], dim=-1)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        # Recompute x, B, C
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                       conv1d_weight, conv1d_bias, seq_idx, None, None, True if ctx.activation in ["silu", "swish"] else False),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)
        
        # Center C to match the centered C from forward pass
        # C = C - C.mean(dim=-1, keepdim=True)
        
        dzxbcdt = torch.empty_like(zxbcdt)
        dzx0, dz, dxBC_given, ddt_given = torch.split(dzxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
        dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)
        if outproj_weight is not None:
            dout_og = dout
            dout = F.linear(dout, outproj_weight.t())
        if d_nonssm > 0:
            dout0, dout = dout.split([d_nonssm, dim], dim=-1)
            if ctx.activation in ["silu", "swish"]:
                _swiglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
            elif ctx.activation == "relu":
                _reglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
        
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        if rmsnorm_weight is None:
            dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
            dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states, *rest = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, dz=dz, recompute_output=recompute_output, output_activation=ctx.output_activation, threshold=ctx.threshold
            )
            out_for_linear = rearrange(rest[0], "b s h p -> b s (h p)") if recompute_output else None
            drmsnorm_weight = None
        else:
            batch = dout.shape[0]
            dy_rms = rearrange(dout, "b s h p -> (b s) (h p)")
            dz = rearrange(dz, "b l d -> (b l) d")
            x_rms = rearrange(out, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            out1_recompute = rearrange(out1_recompute, "b s d -> (b s) d") if recompute_output else None
            dout, drmsnorm_weight, _, dz, *rest = _layer_norm_bwd(dy_rms, x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, None, rstd, z_rms, group_size=dim//ctx.ngroups, norm_before_gate=ctx.norm_before_gate, is_rms_norm=True, recompute_output=recompute_output, dz=dz, out=out1_recompute if recompute_output else None)
            out_for_linear = out_recompute if recompute_output else None
            dout = rearrange(dout, "(b s) (h p) -> b s h p", b=batch, p=headdim)
            dx, ddt, dA, dB, dC, dD, _, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, output_activation=ctx.output_activation, threshold=ctx.threshold
            )

        # Backpropagate through the centering operation: dC w.r.t. uncentered C
        # dC = dC - dC.mean(dim=-1, keepdim=True)
        
        # Copy gradients back into dxBC buffer
        # dxBC.copy_(torch.cat([
        #     rearrange(dx, "b l h p -> b l (h p)"),
        #     rearrange(dB, "b l g n -> b l (g n)"),
        #     rearrange(dC, "b l g n -> b l (g n)")
        # ], dim=-1))
        
        if outproj_weight is not None:
            doutproj_weight = torch.einsum("bso,bsd->od", dout_og, out_for_linear)
            doutproj_bias = dout_og.sum(dim=(0, 1)) if outproj_bias is not None else None
        else:
            doutproj_weight, doutproj_bias = None, None
        dxBC_given = rearrange(dxBC_given, "b s d -> b d s")
        dxBC_given_update, dweight, dbias, *_ = causal_conv1d_bwd_function(
            rearrange_and_update_stride(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias,
            rearrange(dxBC, "b s d -> b d s"), seq_idx, None, None, rearrange_and_update_stride(dxBC_given), False, True if ctx.activation in ["silu", "swish"] else False
        )
        if dxBC_given.stride() != dxBC_given_update.stride():
            dxBC_given.copy_(dxBC_given_update)
        else:
            dxBC_given = dxBC_given_update
        dxBC_given = rearrange(dxBC_given, "b d s -> b s d")
        return dzxbcdt, dweight, dbias, ddt_bias, dA, dD, None, dinitial_states, None, None, None, None, drmsnorm_weight, None, doutproj_weight, doutproj_bias, None, None, None, None, None, None


def sfa_mamba_split_conv1d_scan_combined(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True, return_activation_sparsity = False, output_activation = None, threshold = None):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    return SfaMambaSplitConv1dScanCombinedFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx, dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate, return_activation_sparsity, output_activation, threshold)
