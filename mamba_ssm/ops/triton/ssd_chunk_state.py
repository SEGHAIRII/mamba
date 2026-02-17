# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.ops.triton.softplus import softplus


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


# ============================================================================
# Precompute kernel for DR_SSM: computes x_masked, M, sigmoid_delta, E_mean
# This eliminates redundant computation in all downstream kernels
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=8),
    ],
    key=['seqlen', 'headdim'],
)
@triton.jit
def _precompute_mask_kernel(
    # Input pointers
    E_ptr, I_ptr, x_ptr,
    # Output pointers
    x_masked_ptr, M_ptr, sigmoid_delta_ptr, E_mean_ptr, sigma_squared_sum_ptr,
    # Dimensions
    batch, seqlen, nheads, headdim,
    # Strides for E (batch, seqlen, nheads, headdim)
    stride_E_batch, stride_E_seqlen, stride_E_head, stride_E_hdim,
    # Strides for I (batch, seqlen, nheads, headdim)  
    stride_I_batch, stride_I_seqlen, stride_I_head, stride_I_hdim,
    # Strides for x (batch, seqlen, nheads, headdim)
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # Strides for outputs (same layout as x)
    stride_xm_batch, stride_xm_seqlen, stride_xm_head, stride_xm_hdim,
    stride_M_batch, stride_M_seqlen, stride_M_head, stride_M_hdim,
    stride_sd_batch, stride_sd_seqlen, stride_sd_head, stride_sd_hdim,
    # Strides for E_mean (batch, seqlen, nheads)
    stride_Em_batch, stride_Em_seqlen, stride_Em_head,
    # Strides for sigma_squared_sum (batch, nheads) - for regularization
    stride_sss_batch, stride_sss_head,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # sequence positions per block
    BLOCK_SIZE_N: tl.constexpr,  # headdim elements per block
):
    """
    Precompute mask and masked input for DR_SSM.
    
    For each position (b, t, h, d):
        E_mean[b,t,h] = mean(E[b,t,h,:])
        delta[b,t,h,d] = E[b,t,h,d] - I[b,t,h,d] - E_mean[b,t,h]
        sigmoid_delta[b,t,h,d] = sigmoid(delta)
        M[b,t,h,d] = (sigmoid_delta > 0.5) as float
        x_masked[b,t,h,d] = x[b,t,h,d] * M[b,t,h,d]
        
    Also computes sigma_squared_sum for regularization:
        sigma_squared_sum[b,h] = sum over (t,d) of sigmoid_delta^2
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)  # sequence block
    
    # Offset pointers for this batch and head
    E_ptr += pid_b * stride_E_batch + pid_h * stride_E_head
    I_ptr += pid_b * stride_I_batch + pid_h * stride_I_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    x_masked_ptr += pid_b * stride_xm_batch + pid_h * stride_xm_head
    M_ptr += pid_b * stride_M_batch + pid_h * stride_M_head
    sigmoid_delta_ptr += pid_b * stride_sd_batch + pid_h * stride_sd_head
    E_mean_ptr += pid_b * stride_Em_batch + pid_h * stride_Em_head
    
    # Sequence positions for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    seq_mask = offs_m < seqlen
    
    # Accumulator for sigma_squared_sum (partial sum for this block)
    sigma_sq_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Accumulator for E_sum (must be initialized before loop for Triton)
    E_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Process all headdim elements in chunks - first pass to compute E_mean
    for n_start in range(0, headdim, BLOCK_SIZE_N):
        offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
        hdim_mask = offs_n < headdim
        mask = seq_mask[:, None] & hdim_mask[None, :]
        
        # Load E
        E_ptrs = E_ptr + offs_m[:, None] * stride_E_seqlen + offs_n[None, :] * stride_E_hdim
        E_val = tl.load(E_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # Accumulate E_sum for E_mean computation
        E_sum += tl.sum(E_val, axis=1)
    
    # Compute final E_mean
    E_mean_val = E_sum / headdim
    
    # Store E_mean
    E_mean_ptrs = E_mean_ptr + offs_m * stride_Em_seqlen
    tl.store(E_mean_ptrs, E_mean_val, mask=seq_mask)
    
    # Second pass: compute delta, sigmoid_delta, M, x_masked using computed E_mean
    for n_start in range(0, headdim, BLOCK_SIZE_N):
        offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
        hdim_mask = offs_n < headdim
        mask = seq_mask[:, None] & hdim_mask[None, :]
        
        # Reload E, I, x
        E_ptrs = E_ptr + offs_m[:, None] * stride_E_seqlen + offs_n[None, :] * stride_E_hdim
        I_ptrs = I_ptr + offs_m[:, None] * stride_I_seqlen + offs_n[None, :] * stride_I_hdim
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
        
        E_val = tl.load(E_ptrs, mask=mask, other=0.0).to(tl.float32)
        I_val = tl.load(I_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_val = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute delta = E - I - E_mean
        delta = E_val - I_val - E_mean_val[:, None]
        
        # Compute sigmoid(delta)
        sigmoid_delta = 1.0 / (1.0 + tl.exp(-delta))
        
        # Compute binary mask M = (sigmoid_delta > 0.5)
        M_val = (sigmoid_delta > 0.5).to(x_val.dtype)
        
        # Compute x_masked = x * M
        x_masked_val = x_val * M_val
        
        # Accumulate sigma_squared_sum
        sigma_sq_acc += tl.sum(sigmoid_delta * sigmoid_delta, axis=1)
        
        # Store outputs
        xm_ptrs = x_masked_ptr + offs_m[:, None] * stride_xm_seqlen + offs_n[None, :] * stride_xm_hdim
        M_ptrs = M_ptr + offs_m[:, None] * stride_M_seqlen + offs_n[None, :] * stride_M_hdim
        sd_ptrs = sigmoid_delta_ptr + offs_m[:, None] * stride_sd_seqlen + offs_n[None, :] * stride_sd_hdim
        
        tl.store(xm_ptrs, x_masked_val.to(x_masked_ptr.dtype.element_ty), mask=mask)
        tl.store(M_ptrs, M_val.to(M_ptr.dtype.element_ty), mask=mask)
        tl.store(sd_ptrs, sigmoid_delta.to(sigmoid_delta_ptr.dtype.element_ty), mask=mask)
    
    # Atomic add partial sigma_squared_sum to global accumulator
    sss_ptr = sigma_squared_sum_ptr + pid_b * stride_sss_batch + pid_h * stride_sss_head
    total_sigma_sq = tl.sum(sigma_sq_acc)
    tl.atomic_add(sss_ptr, total_sigma_sq)


def _precompute_mask(E, I, x):
    """
    Precompute mask and masked input for DR_SSM.
    
    Args:
        E: (batch, seqlen, nheads, headdim) - excitation signal
        I: (batch, seqlen, nheads, headdim) - inhibition signal  
        x: (batch, seqlen, nheads, headdim) - input tensor
        
    Returns:
        x_masked: (batch, seqlen, nheads, headdim) - masked input (x * M)
        M: (batch, seqlen, nheads, headdim) - binary mask
        sigmoid_delta: (batch, seqlen, nheads, headdim) - sigmoid values for backward
        E_mean: (batch, seqlen, nheads) - mean of E over headdim
        sigma_squared_sum: (batch, nheads) - sum of sigmoid_delta^2 for regularization
    """
    batch, seqlen, nheads, headdim = E.shape
    assert I.shape == E.shape
    assert x.shape == E.shape
    
    # Allocate outputs
    x_masked = torch.empty_like(x)
    M = torch.empty_like(x)
    sigmoid_delta = torch.empty(batch, seqlen, nheads, headdim, device=E.device, dtype=torch.float32)
    E_mean = torch.empty(batch, seqlen, nheads, device=E.device, dtype=torch.float32)
    sigma_squared_sum = torch.zeros(batch, nheads, device=E.device, dtype=torch.float32)
    
    # Launch kernel: grid over (batch, nheads, seq_blocks)
    grid = lambda META: (
        batch,
        nheads,
        triton.cdiv(seqlen, META['BLOCK_SIZE_M']),
    )
    
    with torch.cuda.device(E.device.index):
        _precompute_mask_kernel[grid](
            E, I, x,
            x_masked, M, sigmoid_delta, E_mean, sigma_squared_sum,
            batch, seqlen, nheads, headdim,
            # E strides
            E.stride(0), E.stride(1), E.stride(2), E.stride(3),
            # I strides
            I.stride(0), I.stride(1), I.stride(2), I.stride(3),
            # x strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            # x_masked strides
            x_masked.stride(0), x_masked.stride(1), x_masked.stride(2), x_masked.stride(3),
            # M strides
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            # sigmoid_delta strides
            sigmoid_delta.stride(0), sigmoid_delta.stride(1), sigmoid_delta.stride(2), sigmoid_delta.stride(3),
            # E_mean strides
            E_mean.stride(0), E_mean.stride(1), E_mean.stride(2),
            # sigma_squared_sum strides
            sigma_squared_sum.stride(0), sigma_squared_sum.stride(1),
        )
    
    return x_masked, M, sigmoid_delta, E_mean, sigma_squared_sum


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}),
        triton.Config({'BLOCK_SIZE_H': 2}),
        triton.Config({'BLOCK_SIZE_H': 4}),
        triton.Config({'BLOCK_SIZE_H': 8}),
        triton.Config({'BLOCK_SIZE_H': 16}),
        triton.Config({'BLOCK_SIZE_H': 32}),
        triton.Config({'BLOCK_SIZE_H': 64}),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr,
    # Matrix dimension
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch, stride_dt_out_chunk, stride_dt_out_head, stride_dt_out_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 2}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 4}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 8}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 16}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 32}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 64}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_cumsum_bwd_kernel(
    # Pointers to matrices
    ddA_ptr, ddt_out_ptr, dt_ptr, A_ptr, dt_bias_ptr,
    ddt_ptr, dA_ptr, ddt_bias_ptr,
    # Matrix dimensions
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_ddA_batch, stride_ddA_chunk, stride_ddA_head, stride_ddA_csize,
    stride_ddt_out_batch, stride_ddt_out_chunk, stride_ddt_out_head, stride_ddt_out_csize,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_ddt_batch, stride_ddt_seqlen, stride_ddt_head,
    stride_dA_head,
    stride_ddt_bias_head,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    ddt_out_ptr += pid_b * stride_ddt_out_batch + pid_c * stride_ddt_out_chunk
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * stride_ddA_chunk
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * chunk_size * stride_ddt_seqlen

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    ddt_out_ptrs = ddt_out_ptr + (offs_h[:, None] * stride_ddt_out_head + offs_c[None, :] * stride_ddt_out_csize)
    ddA_ptrs = ddA_ptr + (offs_h[:, None] * stride_ddA_head + offs_c[None, :] * stride_ddA_csize)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    ddt_ptrs = ddt_ptr + (offs_h[:, None] * stride_ddt_head + offs_c[None, :] * stride_ddt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    ddA = tl.load(ddA_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    ddt_out = tl.load(ddt_out_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    ddt = ddA * A[:, None] + ddt_out
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt_presoftplus = dt
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    clamp_mask = (dt < dt_min) | (dt > dt_max)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    ddt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), ddt, 0.0)
    ddt = tl.where(clamp_mask, 0.0, ddt)
    if DT_SOFTPLUS:
        ddt = tl.where(dt_presoftplus <= 20.0, ddt * tl.sigmoid(dt_presoftplus), ddt)
    tl.store(ddt_ptrs, ddt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit))
    dA = tl.sum(ddA * dt, axis=1)
    tl.atomic_add(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    if HAS_DT_BIAS:
        ddt_bias = tl.sum(ddt, axis=1)
        tl.atomic_add(ddt_bias_ptr + offs_h * stride_ddt_bias_head, ddt_bias, mask=offs_h < nheads)




@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, states_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            # scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
            scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k
        else:
            # scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
            scale = tl.where((seq_idx_last >= 0) & (seq_idx_k == seq_idx_last), tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, dstates_ptr, dt_ptr, dA_cumsum_ptr,
    dx_ptr, ddt_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_states_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_k[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_hdim + offs_k[:, None] * stride_states_dstate)
    if BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_states_dstate

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # acc *= tl.exp(dA_cs_last - dA_cs_m)[:, None]
    acc *= tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))[:, None]

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)
    ddA_cs = -(ddt * dt_m)
    ddA_cs_last = -tl.sum(ddA_cs)
    ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
    tl.atomic_add(ddA_cumsum_ptr + (chunk_size - 1) * stride_ddA_cs_csize, ddA_cs_last)

    dx = (acc * dt_m[:, None]).to(dx_ptr.dtype.element_ty)
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'dstate', 'hdim'],
)
@triton.jit
def _chunk_state_bwd_db_kernel(
    # Pointers to matrices
    x_ptr, dstates_ptr, b_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    db_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, dstate, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_db_batch, stride_db_seqlen, stride_db_split, stride_db_group, stride_db_dstate,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_DDA_CS: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_x_head
    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_g * stride_db_group + pid_s * stride_db_split
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_states_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dA_cs_head
    if HAS_DDA_CS:
        b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_g * stride_b_head
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_ddA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_k[None, :] * stride_x_hdim)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_dstate + offs_k[:, None] * stride_states_hdim)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    if HAS_DDA_CS:
        b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_n[None, :] * stride_b_dstate)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    for h in range(nheads_iter):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0)
        dstates = dstates.to(x_ptrs.dtype.element_ty)
        db = tl.dot(x, dstates)
        dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            # scale = tl.exp(dA_cs_last - dA_cs_m)
            scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
        else:
            # scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
            scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)), 0.0)
        db *= (scale * dt_m)[:, None]
        if HAS_DDA_CS:
            # This is the gradient wrt (dA_cs_last - dA_cs_m), i.e. the exclusive reverse cumsum
            ddA_cs = tl.sum(db * b, axis=1)
            tl.atomic_add(ddA_cumsum_ptrs + stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size - 1)
        acc += db
        x_ptrs += stride_x_head
        dstates_ptrs += stride_states_head
        dt_ptrs += stride_dt_head
        dA_cumsum_ptr += stride_dA_cs_head
        dA_cumsum_ptrs += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # if HAS_SEQ_IDX:
    #     seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
    #     seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    #     acc = tl.where(seq_idx_m[:, None] == seq_idx_last, acc, 0.0)
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_dstate)
    tl.store(db_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_state_bwd_ddAcs_stable_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, dstates_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_states_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_k[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_hdim + offs_k[:, None] * stride_states_dstate)
    if BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_states_dstate

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    if not HAS_SEQ_IDX:
        # scale = tl.exp(dA_cs_last - dA_cs_m)
        scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        # scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)), 0.0)
    acc *= scale[:, None]

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    ddt = tl.sum(acc * x, axis=1)
    # ddA_cs = -(ddt * dt_m)
    # Triton 2.2.0 errors if we have the cumsum here, so we just write it out
    # then call torch.cumsum outside this kernel.
    # ddA_cs = tl.cumsum(ddt * dt_m)
    ddA_cs = ddt * dt_m
    ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    # tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
    tl.atomic_add(ddA_cumsum_ptrs + stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size - 1)


# =================================================== dr
# dr - optimized version using precomputed x_masked
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _chunk_state_fwd_dr_kernel(
    # Pointers to matrices - now uses x_masked instead of x, E, I, E_mean
    x_masked_ptr, b_ptr, states_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides for x_masked (batch, seqlen, nheads, headdim)
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # Strides for b (batch, seqlen, ngroups, dstate)
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    # Strides for states (batch, nchunks, nheads, headdim, dstate)
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    # Strides for dA_cumsum (batch, nheads, nchunks, chunk_size, headdim)
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
    # Strides for seq_idx
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    
    # Setup pointers - now only x_masked and b, no E, I, E_mean
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_masked_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # x_masked is already precomputed with mask applied
    x_ptrs = x_masked_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize + offs_m * stride_dA_cs_hdim,
                         mask=(offs_m < hdim), other=0.0).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k[None, :] * stride_dA_cs_csize + offs_m[:, None] * stride_dA_cs_hdim
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen
    
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop - no mask computation needed, x_masked already has mask applied
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):  
        # Load precomputed x_masked (already x * M)
        x_masked = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
        
        # Calculate scale with headdim-specific A: shape (M, K)
        if not HAS_SEQ_IDX:
            scale = tl.exp(tl.minimum((dA_cs_last[:, None] - dA_cs_k), 0.0))
        else:
            scale = tl.where((seq_idx_last >= 0) & (seq_idx_k == seq_idx_last), 
                           tl.exp(tl.minimum((dA_cs_last[:, None] - dA_cs_k), 0.0)), 0.0)

        x_scaled = x_masked * scale
        b = b.to(x_masked_ptr.dtype.element_ty)
        acc += tl.dot(x_scaled.to(x_masked_ptr.dtype.element_ty), b)
      
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize   
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    
    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)
    

#dr
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero([ "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero([ "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero([ "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero([ "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero([ "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero([ "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero([ "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero([ "ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_state_bwd_dx_dr_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, E_ptr, I_ptr, E_mean_ptr, dstates_ptr, dA_cumsum_ptr, seq_idx_ptr,
    dx_ptr, ddA_cumsum_ptr, 
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_E_batch, stride_E_seqlen, stride_E_head, stride_E_hdim,
    stride_I_batch, stride_I_seqlen, stride_I_head, stride_I_hdim,
    stride_E_mean_batch, stride_E_mean_seqlen, stride_E_mean_head,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize, stride_ddA_cs_hdim,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_states_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    E_ptr += pid_b * stride_E_batch + pid_c * chunk_size * stride_E_seqlen + pid_h * stride_E_head
    I_ptr += pid_b * stride_I_batch + pid_c * chunk_size * stride_I_seqlen + pid_h * stride_I_head
    E_mean_ptr += pid_b * stride_E_mean_batch + pid_c * chunk_size * stride_E_mean_seqlen + pid_h * stride_E_mean_head
    
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_k[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_hdim + offs_k[:, None] * stride_states_dstate)
    
    if BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_states_dstate


    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load per-headdim dA_cumsum: dA_cs_last has shape (N,), dA_cs has shape (M, N)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize + offs_n * stride_dA_cs_hdim,
                        mask=offs_n < hdim, other=0.0).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m[:, None] * stride_dA_cs_csize + offs_n[None, :] * stride_dA_cs_hdim
    dA_cs_mn = tl.load(dA_cumsum_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)

    # Compute scale: shape (M, N)
    scale = tl.exp(tl.minimum(dA_cs_last[None, :] - dA_cs_mn, 0.0))

    # Load E, I, E_mean and compute mask
    E_ptrs = E_ptr + offs_m[:, None] * stride_E_seqlen + offs_n[None, :] * stride_E_hdim
    I_ptrs = I_ptr + offs_m[:, None] * stride_I_seqlen + offs_n[None, :] * stride_I_hdim
    E_mean_ptrs = E_mean_ptr + offs_m * stride_E_mean_seqlen

    E = tl.load(E_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    I = tl.load(I_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    E_mean = tl.load(E_mean_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)

    sigma = E - I - E_mean[:, None]
    mask = (tl.sigmoid(sigma) > 0.5).to(tl.float32)

    # Apply scale and mask to acc
    acc = acc * scale * mask

    # Compute ddA_cumsum gradient
    # d(scale)/d(dA_cs_mn) = -scale, d(scale)/d(dA_cs_last) = +scale
    # intermediate = acc_before_scale * mask (but we need original acc before scale was applied)
    # So: d_loss/d_scale = acc_before_scale_mask * (dstates @ B^T result)
    # Actually we need x to compute ddA_cs properly

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)

    # acc at this point = (dstates @ B^T) * scale * mask
    # For ddA_cs: we need (dstates @ B^T) * x * mask * scale = acc * x / scale * scale = acc * x
    # But more precisely: d_loss = (dstates @ B^T) * scale * mask, and d(scale) = scale * d(dA_cs_last - dA_cs_mn)
    # So ddA_cs contribution = acc * x (before mask application would give different result)
    # Let's compute fresh: intermediate = (dstates @ B^T) / (scale * mask) * scale * mask * x = (dstates @ B^T) * x
    # Actually acc already has scale*mask, so: acc / (scale * mask) = original dstates @ B^T
    # Then: ddA_contribution = (original) * x * mask * scale = acc * x

    ddA_intermediate = acc * x  # shape (M, N)
    ddA_cs_mn = -ddA_intermediate  # gradient for dA_cs at position (m, n)
    ddA_cs_last_contribution = tl.sum(ddA_intermediate, axis=0)  # sum over M, shape (N,)

    # Atomic add for ddA_cumsum (per headdim)
    ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m[:, None] * stride_ddA_cs_csize + offs_n[None, :] * stride_dA_cs_hdim
    tl.atomic_add(ddA_cumsum_ptrs, ddA_cs_mn, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < hdim))
    ddA_cs_last_ptrs = ddA_cumsum_ptr + (chunk_size - 1) * stride_ddA_cs_csize + offs_n * stride_dA_cs_hdim
    tl.atomic_add(ddA_cs_last_ptrs, ddA_cs_last_contribution, mask=offs_n < hdim)

    # Store dx
    dx = acc.to(dx_ptr.dtype.element_ty)
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))


#dr
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
    ],
    key=['chunk_size', 'dstate', 'hdim', 'HAS_DDA_CS'],
)
@triton.jit
def _chunk_state_bwd_db_dr_kernel(
    # Inputs - using precomputed x_masked and M
    x_masked_ptr, M_ptr, dstates_ptr, B_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Outputs
    dB_ptr, ddA_next_ptr,
    # Dimensions
    chunk_size, hdim, dstate, batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides for x_masked (b, l, h, p)
    stride_xm_batch, stride_xm_seqlen, stride_xm_head, stride_xm_hdim,
    # Strides for M (b, l, h, p)
    stride_M_batch, stride_M_seqlen, stride_M_head, stride_M_hdim,
    # Strides for dstates (b, c, h, p, n)
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    # Strides for B (b, l, g, n)
    stride_B_batch, stride_B_seqlen, stride_B_head, stride_B_dstate,
    # Strides for dA_cumsum (b, h, c, l, p)
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
    # Strides for output dB (b, l, nsplits, g, n)
    stride_dB_batch, stride_dB_seqlen, stride_dB_split, stride_dB_group, stride_dB_dstate,
    # Strides for output ddA_next (b, l, h, p) - per-channel
    stride_ddA_batch, stride_ddA_seqlen, stride_ddA_head, stride_ddA_hdim,
    # Seq idx strides
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta
    HAS_DDA_CS: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups  # which split of heads
    pid_g = pid_sg - pid_s * ngroups  # which group
    
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n  # time block
    pid_n = tl.program_id(axis=0) % num_pid_n   # dstate block
    
    # Tile offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # time in chunk
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # dstate
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # headdim (reduction)
    
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    # Output pointers for dB (per-split accumulator)
    dB_ptr += pid_b * stride_dB_batch + pid_c * chunk_size * stride_dB_seqlen + pid_g * stride_dB_group + pid_s * stride_dB_split
    dB_ptrs = dB_ptr + (offs_m[:, None] * stride_dB_seqlen + offs_n[None, :] * stride_dB_dstate)
    
    # ddA_next pointer (if computing)
    if HAS_DDA_CS:
        ddA_next_ptr += pid_b * stride_ddA_batch + pid_c * chunk_size * stride_ddA_seqlen
        # Will index by head inside loop
    
    # Input pointers setup (offset by batch and chunk) - using precomputed x_masked and M
    base_xm_ptr = x_masked_ptr + pid_b * stride_xm_batch + pid_c * chunk_size * stride_xm_seqlen
    base_M_ptr = M_ptr + pid_b * stride_M_batch + pid_c * chunk_size * stride_M_seqlen
    base_B_ptr = B_ptr + pid_b * stride_B_batch + pid_c * chunk_size * stride_B_seqlen + pid_g * stride_B_head
    base_dA_cs_ptr = dA_cumsum_ptr + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk
    
    # dstates for this chunk (per head)
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk
    
    # Accumulators
    acc_dB = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        acc_ddA = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)  # per headdim
    
    # Load decay factor: dA_cumsum at last position minus current
    # dA_cs_end shape: (nheads, hdim)
    dA_cs_end = tl.load(base_dA_cs_ptr + (chunk_size - 1) * stride_dA_cs_csize + offs_k * stride_dA_cs_hdim, 
                        mask=offs_k < hdim, other=0.0).to(tl.float32)
    
    # Iterate over heads assigned to this program
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    
    for h_iter in range(nheads_iter):
        h = pid_g * (nheads // ngroups) + pid_s * nheads_per_program + h_iter
        
        # Note: can't use break in Triton, but nheads_iter is already bounded correctly
        # so h < nheads is guaranteed by the min() above
            
        # Pointers for this head - using precomputed x_masked and M
        xm_ptrs = base_xm_ptr + h * stride_xm_head + (offs_m[:, None] * stride_xm_seqlen + offs_k[None, :] * stride_xm_hdim)
        M_ptrs = base_M_ptr + h * stride_M_head + (offs_m[:, None] * stride_M_seqlen + offs_k[None, :] * stride_M_hdim)
        
        # dstates for this head: (hdim, dstate)
        dstates_ptrs = dstates_ptr + h * stride_dstates_head + (offs_k[:, None] * stride_dstates_hdim + offs_n[None, :] * stride_dstates_dstate)
        
        # ddA pointer for this head
        if HAS_DDA_CS:
            ddA_ptrs_h = ddA_next_ptr + h * stride_ddA_head + (offs_m[:, None] * stride_ddA_seqlen + offs_k[None, :] * stride_ddA_hdim)
        
        # Reduction loop over headdim blocks
        for k_start in range(0, hdim, BLOCK_SIZE_K):
            k = k_start + offs_k
            mask_k = k < hdim
            
            # Load precomputed x_masked directly
            x_masked = tl.load(xm_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & mask_k[None, :], other=0.0).to(tl.float32)
            
            # Load current position dA_cumsum for decay calculation
            dA_cs_m = tl.load(base_dA_cs_ptr + h * stride_dA_cs_head + offs_m[:, None] * stride_dA_cs_csize + k[None, :] * stride_dA_cs_hdim,
                              mask=(offs_m[:, None] < chunk_size_limit) & mask_k[None, :], other=0.0).to(tl.float32)
            
            # Decay: exp(dA_end - dA_m)
            decay = tl.exp(tl.minimum(dA_cs_end - dA_cs_m, 0.0))
            
            # Apply decay to masked X
            x_scaled = x_masked * decay  # (BLOCK_M, BLOCK_K)
            
            # Load dstates: (BLOCK_K, BLOCK_N)
            dstates = tl.load(dstates_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < dstate), other=0.0)
            dstates = dstates.to(tl.float32)  # Cast to float32 to match x_scaled
            
            # Accumulate dB: x_scaled^T @ dstates -> wait, dimensions?
            # x_scaled is (time, hdim), dstates is (hdim, dstate)
            # We want dB[time, dstate] += sum_hdim(x_scaled[time, hdim] * dstates[hdim, dstate])
            # That's a direct dot product: acc += x_scaled \cdot dstates (batched over time,dstate via hdim)
            acc_dB += tl.dot(x_scaled, dstates)
            
            # Compute ddA_next contribution if needed
            if HAS_DDA_CS:
                # Load B for this time and dstate: (BLOCK_M, BLOCK_N)
                B_load_ptrs = base_B_ptr + (offs_m[:, None] * stride_B_seqlen + offs_n[None, :] * stride_B_dstate)
                B_val = tl.load(B_load_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
                
                # inner_prod = sum_n(B[time,n] * dstates[hdim,n]) 
                # Result shape: (time, hdim)
                
                # Transpose dstates to (dstate, hdim) implicitly by dot
                inner = tl.dot(B_val, tl.trans(dstates))  # (BLOCK_M, BLOCK_K)
                contrib = -x_masked * decay * inner
                
                # Accumulate to ddA
                tl.atomic_add(ddA_ptrs_h + k_start, contrib, mask=(offs_m[:, None] < chunk_size_limit) & mask_k[None, :])
            
            # Increment pointers for next headdim block
            xm_ptrs += BLOCK_SIZE_K * stride_xm_hdim
            M_ptrs += BLOCK_SIZE_K * stride_M_hdim
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_hdim
    
    # Store dB
    tl.store(dB_ptrs, acc_dB.to(dB_ptr.dtype.element_ty), mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))




#dr
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_state_bwd_dE_dI_dEmean_dr_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, dstates_ptr, dA_cumsum_ptr, seq_idx_ptr,
    E_ptr, I_ptr, E_mean_ptr,
    dE_ptr, dI_ptr, dE_mean_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_E_batch, stride_E_seqlen, stride_E_head, stride_E_hdim,
    stride_I_batch, stride_I_seqlen, stride_I_head, stride_I_hdim,
    stride_E_mean_batch, stride_E_mean_seqlen, stride_E_mean_head,
    stride_dE_batch, stride_dE_seqlen, stride_dE_head, stride_dE_hdim,
    stride_dI_batch, stride_dI_seqlen, stride_dI_head, stride_dI_hdim,
    stride_dE_mean_batch, stride_dE_mean_seqlen, stride_dE_mean_head,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    
    # Pointer setup
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_states_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    E_ptr += pid_b * stride_E_batch + pid_c * chunk_size * stride_E_seqlen + pid_h * stride_E_head
    I_ptr += pid_b * stride_I_batch + pid_c * chunk_size * stride_I_seqlen + pid_h * stride_I_head
    E_mean_ptr += pid_b * stride_E_mean_batch + pid_c * chunk_size * stride_E_mean_seqlen + pid_h * stride_E_mean_head
    dE_ptr += pid_b * stride_dE_batch + pid_c * chunk_size * stride_dE_seqlen + pid_h * stride_dE_head
    dI_ptr += pid_b * stride_dI_batch + pid_c * chunk_size * stride_dI_seqlen + pid_h * stride_dI_head
    dE_mean_ptr += pid_b * stride_dE_mean_batch + pid_c * chunk_size * stride_dE_mean_seqlen + pid_h * stride_dE_mean_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    # Matmul: acc = B @ dstates = (dstates @ B^T)^T, but we compute B @ dstates directly
    # This gives (dstates @ B^T) result with shape (chunk_size, hdim) = (M, N)
    offs_k = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_k[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_hdim + offs_k[:, None] * stride_states_dstate)
    
    if BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_states_dstate

    # Recompute offsets after loop
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load per-headdim dA_cumsum
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize + offs_n * stride_dA_cs_hdim,
                         mask=offs_n < hdim, other=0.0).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m[:, None] * stride_dA_cs_csize + offs_n[None, :] * stride_dA_cs_hdim
    dA_cs_mn = tl.load(dA_cumsum_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    
    if not HAS_SEQ_IDX:
        scale = tl.exp(tl.minimum(dA_cs_last[None, :] - dA_cs_mn, 0.0))
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_m[:, None] == seq_idx_last, 
                        tl.exp(tl.minimum(dA_cs_last[None, :] - dA_cs_mn, 0.0)), 0.0)
    
    acc *= scale  # acc = (dstates @ B^T) * scale, shape (M, N)

    # Load x
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    
    # Load E, I, E_mean
    E_ptrs = E_ptr + (offs_m[:, None] * stride_E_seqlen + offs_n[None, :] * stride_E_hdim)
    I_ptrs = I_ptr + (offs_m[:, None] * stride_I_seqlen + offs_n[None, :] * stride_I_hdim)
    E_mean_in_ptrs = E_mean_ptr + offs_m * stride_E_mean_seqlen
    
    E = tl.load(E_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    I = tl.load(I_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    E_mean_val = tl.load(E_mean_in_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    
    # Compute sigma and sigmoid
    sigma = E - I - E_mean_val[:, None]
    sigmoid_sigma = tl.sigmoid(sigma)
    
    # dmask = (dstates @ B^T) * scale * x
    dmask = acc * x
    
    # dsigma via STE: dsigma = dmask * sigmoid' = dmask * sigmoid * (1 - sigmoid)
    dsigma = dmask * sigmoid_sigma * (1.0 - sigmoid_sigma)
    
    # dE = dsigma
    dE_ptrs = dE_ptr + (offs_m[:, None] * stride_dE_seqlen + offs_n[None, :] * stride_dE_hdim)
    tl.atomic_add(dE_ptrs, dsigma, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    
    # dI = -dsigma
    dI_ptrs = dI_ptr + (offs_m[:, None] * stride_dI_seqlen + offs_n[None, :] * stride_dI_hdim)
    tl.atomic_add(dI_ptrs, -dsigma, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    
    # dE_mean = -sum(dsigma, dim=hdim) for each position m
    dE_mean_contrib = -tl.sum(dsigma, axis=1)  # shape (M,)
    dE_mean_out_ptrs = dE_mean_ptr + offs_m * stride_dE_mean_seqlen
    tl.atomic_add(dE_mean_out_ptrs, dE_mean_contrib, mask=offs_m < chunk_size_limit)

# =============================================================




@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _chunk_state_varlen_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, dt_ptr, dA_cumsum_ptr, chunk_states_ptr, cu_seqlens_ptr, states_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_chunk_states_chunk, stride_chunk_states_head, stride_chunk_states_hdim, stride_chunk_states_dstate,
    stride_states_batch, stride_states_head, stride_states_hdim, stride_states_dstate,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    pid_c = (end_idx - 1) // chunk_size
    b_ptr += pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = end_idx - pid_c * chunk_size
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k) & (offs_k[None, :] >= start_idx_cur - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate) & (offs_k[:, None] >= start_idx_cur - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        # scale = tl.where((offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
        #                  tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
        scale = tl.where((offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
                         tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # If the sequence starts after the last chunk idx, we don't need to add the contribution from the last chunk
    if start_idx < pid_c * chunk_size:
        chunk_states_ptrs = chunk_states_ptr + (offs_m[:, None] * stride_chunk_states_hdim + offs_n[None, :] * stride_chunk_states_dstate)
        chunk_states = tl.load(chunk_states_ptrs, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        # scale = tl.where(start_idx < pid_c * chunk_size, tl.exp(dA_cs_last), 0.0)
        scale = tl.exp(dA_cs_last)
        acc += chunk_states * scale

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)





def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt, A, dt_bias, dt_out, dA_cumsum,
            batch, seqlen, nheads, chunk_size,
            dt_limit[0], dt_limit[1],
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out



def _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")), ddt=None):
    batch, seqlen, nheads = dt.shape
    _, _, nchunks, chunk_size = ddA.shape
    assert ddA.shape == (batch, nheads, nchunks, chunk_size)
    assert ddt_out.shape == (batch, nheads, nchunks, chunk_size)
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
        ddt_bias = torch.empty_like(dt_bias, dtype=torch.float32)
    else:
        ddt_bias = None
    if ddt is not None:
        assert ddt.shape == dt.shape
    else:
        ddt = torch.empty_like(dt)
    dA = torch.empty_like(A, dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_bwd_kernel[grid_chunk_cs](
            ddA, ddt_out, dt, A, dt_bias, ddt, dA, ddt_bias,
            batch, seqlen, nheads, chunk_size,
            dt_limit[0], dt_limit[1],
            ddA.stride(0), ddA.stride(2), ddA.stride(1), ddA.stride(3),
            ddt_out.stride(0), ddt_out.stride(2), ddt_out.stride(1), ddt_out.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            ddt.stride(0), ddt.stride(1), ddt.stride(2),
            dA.stride(0),
            ddt_bias.stride(0) if ddt_bias is not None else 0,
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return ddt, dA, ddt_bias

#dr
def _chunk_cumsum_A_fwd(A, chunk_size):
    
    batch, seqlen, nheads, headdim = A.shape
    nchunks = math.ceil(seqlen / chunk_size)
    A_chunked = A.view(batch, nchunks, chunk_size, nheads, headdim)
    
    # Cumsum along chunk dimension
    dA_cumsum = torch.cumsum(A_chunked, dim=2)  # Sum along chunk_size
    
    # Rearrange to expected output format: (batch, nheads, nchunks, chunk_size, headdim)
    dA_cumsum = dA_cumsum.permute(0, 3, 1, 2, 4).contiguous()
    return dA_cumsum

#dr
def _chunk_cumsum_A_bwd(ddA_cumsum, seqlen):
    """
    Gradient of cumsum is reverse cumsum.
    
    Input:
        ddA_cumsum: (batch, nheads, nchunks, chunk_size, headdim)
        seqlen: original sequence length (before padding)
    Output:
        dA: (batch, seqlen, nheads, headdim)
    """
    batch, nheads, nchunks, chunk_size, headdim = ddA_cumsum.shape
    
    # Rearrange: (batch, nheads, nchunks, chunk_size, headdim) -> (batch, nchunks, chunk_size, nheads, headdim)
    ddA_chunked = ddA_cumsum.permute(0, 2, 3, 1, 4)
    
    # Reverse cumsum: gradient of cumsum at position i is sum of all gradients from i to end
    dA_chunked = torch.flip(torch.cumsum(torch.flip(ddA_chunked, dims=[2]), dim=2), dims=[2])
    
    # Flatten back to sequence: (batch, nchunks, chunk_size, nheads, headdim) -> (batch, nchunks*chunk_size, nheads, headdim)
    dA = dA_chunked.reshape(batch, nchunks * chunk_size, nheads, headdim)
    
    return dA


def _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x, B, states, dt, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return states


def _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if dx is not None:
        assert dx.shape == x.shape
    else:
        dx = torch.empty_like(x)
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dA_cumsum.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                       batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_dx_kernel[grid_dx](
            x, B, dstates, dt, dA_cumsum, dx, ddt, ddA_cumsum,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3),
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        )
    return dx, ddt.to(dt.dtype), ddA_cumsum.to(dA_cumsum.dtype)


def _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=None, B=None, ngroups=1):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B is not None:
        assert B.shape == (batch, seqlen, ngroups, dstate)
        B_strides = (B.stride(0), B.stride(1), B.stride(2), B.stride(3))
        # Use torch.empty since the Triton kernel will call init_to_zero
        ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=x.device, dtype=torch.float32)
        ddA_cumsum_strides = (ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3))
    else:
        B_strides = (0, 0, 0, 0)
        ddA_cumsum = None
        ddA_cumsum_strides = (0, 0, 0, 0)
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dB = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=x.device, dtype=torch.float32)
    grid_db = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                        batch * nchunks, nsplits * ngroups)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_db_kernel[grid_db](
            x, dstates, B, dt, dA_cumsum, seq_idx, dB, ddA_cumsum,
            chunk_size, dstate, headdim,
            batch, seqlen, nheads, nheads_per_program, ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            *B_strides,
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dB.stride(0), dB.stride(1), dB.stride(2), dB.stride(3), dB.stride(4),
            *ddA_cumsum_strides,
            HAS_DDA_CS=ddA_cumsum is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    dB = dB.sum(2)
    if ddA_cumsum is not None:
        # The first element of ddA_cumsum is always zero, since that dA_cumsum does not contribute
        # to the state of the chunk.
        # torch.cumsum(ddA_cumsum[..., 1:], dim=-1, out=ddA_cumsum[..., 1:])
        # But it's easier to just do the cumsum for all elements, the result will be the same.
        torch.cumsum(ddA_cumsum, dim=-1, out=ddA_cumsum)
    return dB if B is None else (dB, ddA_cumsum)


def _chunk_state_bwd_ddAcs_stable(B, x, dt, dA_cumsum, dstates, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # Use torch.empty since the Triton kernel will call init_to_zero
    ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                          batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_ddAcs_stable_kernel[grid_ddtcs](
            x, B, dstates, dt, dA_cumsum, seq_idx, ddA_cumsum,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3),
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_M=max(triton.next_power_of_2(chunk_size), 16),
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        )
    torch.cumsum(ddA_cumsum[..., 1:], dim=-1, out=ddA_cumsum[..., 1:])
    return ddA_cumsum

#dr
def _chunk_state_dr_fwd(B, x_masked, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True):
    """
    Compute chunk states for DR_SSM using precomputed masked input.
    
    Args:
        B: (batch, seqlen, ngroups, dstate) - B matrix
        x_masked: (batch, seqlen, nheads, headdim) - precomputed x * M
        dA_cumsum: (batch, nheads, nchunks, chunk_size, headdim) - cumulative decay
        seq_idx: optional sequence indices
        states: optional preallocated states buffer
        states_in_fp32: whether to use fp32 for states
        
    Returns:
        states: (batch, nchunks, nheads, headdim, dstate) - chunk states
    """
    batch, seqlen, nheads, headdim = x_masked.shape
    _, _, nchunks, chunk_size, _ = dA_cumsum.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size, headdim)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x_masked.device, dtype=states_dtype)
            
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    with torch.cuda.device(x_masked.device.index):
        _chunk_state_fwd_dr_kernel[grid](
            x_masked, B, states, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            # x_masked strides
            x_masked.stride(0), x_masked.stride(1), x_masked.stride(2), x_masked.stride(3),
            # B strides
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            # states strides
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            # dA_cumsum strides
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
            # seq_idx strides
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
        
    return states

#dr
def _chunk_state_bwd_dr_dx(B, x, E, I, E_mean, dA_cumsum, dstates, seq_idx=None, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size, _ = dA_cumsum.shape  # dA_cumsum now has headdim dim
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size, headdim)
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    assert E.shape == (batch, seqlen, nheads, headdim)
    assert I.shape == (batch, seqlen, nheads, headdim)
    assert E_mean.shape == (batch, seqlen, nheads)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if dx is not None:
        assert dx.shape == x.shape
    else:
        dx = torch.empty_like(x)
    ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, headdim, device=dA_cumsum.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                       batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_dx_dr_kernel[grid_dx](
            x, B, E, I, E_mean, dstates, dA_cumsum, seq_idx,
            dx, ddA_cumsum,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            E.stride(0), E.stride(1), E.stride(2), E.stride(3),
            I.stride(0), I.stride(1), I.stride(2), I.stride(3),
            E_mean.stride(0), E_mean.stride(1), E_mean.stride(2),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4),
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        )
    return dx, ddA_cumsum.to(dA_cumsum.dtype)

#dr

def _chunk_state_bwd_db_dr(
    x_masked, M, dstates, B, dA_cumsum, seq_idx=None,
    dB=None, ddA_next=None, has_ddA=False
):
    """
    Compute dB gradient using precomputed x_masked and M.
    """
    batch, seqlen, nheads, headdim = x_masked.shape
    _, nchunks, _, _, dstate = dstates.shape
    _, _, ngroups, _ = B.shape
    
    assert nheads % ngroups == 0
    nheads_ngroups_ratio = nheads // ngroups
    
    if dB is None:
        # Use split to reduce contention
        sm_count = torch.cuda.get_device_properties(x_masked.device).multi_processor_count
        nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
        nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
        dB = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=x_masked.device, dtype=torch.float32)
    else:
        # Assume dB already has correct shape including splits or handle unsqueeze
        if dB.dim() == 4:  # (b, l, g, n)
            dB = dB.unsqueeze(2)  # Add split dim
        nsplits = dB.shape[2]
        nheads_per_program = nheads_ngroups_ratio // nsplits
    
    if has_ddA:
        if ddA_next is None:
            ddA_next = torch.empty(batch, seqlen, nheads, headdim, device=x_masked.device, dtype=torch.float32)
        else:
            assert ddA_next.shape == (batch, seqlen, nheads, headdim)
    
    # Rearrange to chunk format if needed, or assume already chunked
    # Assuming inputs are already in (b, l, ...) format, kernel handles chunk indexing via strides
    
    grid = lambda META: (
        triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
        batch * nchunks,
        nsplits * ngroups
    )
    
    # Get chunk_size from dA_cumsum
    chunk_size = dA_cumsum.shape[3]
    
    with torch.cuda.device(x_masked.device.index):
        _chunk_state_bwd_db_dr_kernel[grid](
            x_masked, M, dstates, B, dA_cumsum, seq_idx,
            dB, ddA_next,
            chunk_size, headdim, dstate, batch, seqlen, nheads, nheads_per_program, ngroups,
            # x_masked strides
            x_masked.stride(0), x_masked.stride(1), x_masked.stride(2), x_masked.stride(3),
            # M strides
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            # dstates strides (b, c, h, p, n)
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            # B strides (b, l, g, n)
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            # dA_cumsum strides (b, h, c, l, p)
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
            # dB strides (b, l, s, g, n)
            dB.stride(0), dB.stride(1), dB.stride(2), dB.stride(3), dB.stride(4),
            # ddA strides (b, l, h, p) or zeros
            *((ddA_next.stride(0), ddA_next.stride(1), ddA_next.stride(2), ddA_next.stride(3)) if has_ddA else (0,0,0,0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_DDA_CS=has_ddA,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    
    # Sum over splits to get final dB
    dB = dB.sum(dim=2)  # Remove split dimension
    
    return dB, ddA_next



#dr
def _chunk_state_bwd_dr_dE_dI_dEmean(B, x, E, I, E_mean, dA_cumsum, dstates, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size, _ = dA_cumsum.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size, headdim)
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    assert E.shape == (batch, seqlen, nheads, headdim)
    assert I.shape == (batch, seqlen, nheads, headdim)
    assert E_mean.shape == (batch, seqlen, nheads)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    
    # Initialize outputs to zero for atomic adds
    dE = torch.zeros_like(E, dtype=torch.float32)
    dI = torch.zeros_like(I, dtype=torch.float32)
    dE_mean = torch.zeros_like(E_mean, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                          batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_dE_dI_dEmean_dr_kernel[grid](
            x, B, dstates, dA_cumsum, seq_idx,
            E, I, E_mean,
            dE, dI, dE_mean,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            E.stride(0), E.stride(1), E.stride(2), E.stride(3),
            I.stride(0), I.stride(1), I.stride(2), I.stride(3),
            E_mean.stride(0), E_mean.stride(1), E_mean.stride(2),
            dE.stride(0), dE.stride(1), dE.stride(2), dE.stride(3),
            dI.stride(0), dI.stride(1), dI.stride(2), dI.stride(3),
            dE_mean.stride(0), dE_mean.stride(1), dE_mean.stride(2),
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_M=max(triton.next_power_of_2(chunk_size), 16),
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        )
    
    
    dE += dE_mean / headdim
    return dE.to(E.dtype), dI.to(I.dtype)




def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    assert nheads % ngroups == 0
    assert B.shape == (total_seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)
    states = torch.empty(batch, nheads, headdim, dstate, dtype=chunk_states.dtype, device=chunk_states.device)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_varlen_kernel[grid](
            x, B, dt, dA_cumsum, chunk_states, cu_seqlens, states,
            headdim, dstate, chunk_size,
            total_seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            dt.stride(1), dt.stride(0), dt.stride(2),
            dA_cumsum.stride(1), dA_cumsum.stride(0), dA_cumsum.stride(2),
            chunk_states.stride(0), chunk_states.stride(1), chunk_states.stride(2), chunk_states.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        )
    return states


class ChunkStateFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, x, dt, dA_cumsum, states_in_fp32=True):
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen <= nchunks * chunk_size
        _, _, ngroups, dstate = B.shape
        assert B.shape == (batch, seqlen, ngroups, dstate)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        if B.stride(-1) != 1:
            B = B.contiguous()
        if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
            x = x.contiguous()
        states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=states_in_fp32)
        ctx.save_for_backward(B, x, dt, dA_cumsum)
        return states

    @staticmethod
    def backward(ctx, dstates):
        B, x, dt, dA_cumsum = ctx.saved_tensors
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        _, _, ngroups, dstate = B.shape
        assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
        if dstates.stride(-1) != 1:
            dstates = dstates.contiguous()
        dx, ddt, ddA_cumsum = _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates)
        dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, ngroups=ngroups)
        dB = dB.to(B.dtype)
        return dB, dx, ddt, ddA_cumsum, None


def chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True):
    """
    Argument:
        B: (batch, seqlen, ngroups, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    return ChunkStateFn.apply(B, x, dt, dA_cumsum, states_in_fp32)


def chunk_state_ref(B, x, dt, dA_cumsum):
    """
    Argument:
        B: (batch, seqlen, ngroups, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)
