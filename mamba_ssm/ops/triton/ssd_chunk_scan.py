# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
from packaging import version

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
)
@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, prev_states_ptr, D_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head,
    THRESHOLD: tl.constexpr,

    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    OUTPUT_ACTIVATION: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # With Triton 2.2.0, this works
    if IS_TRITON_22 or pid_c > -1:
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
        prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        # cb *= tl.exp((dA_cs_m[:, None] - dA_cs_k[None, :]))
        cb *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_k[None, :]), 0.0))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim), other=0.0)
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize



    # this is hard coded for now, modify later, relufy y
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)


    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
                             mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc += x_residual * D
        
        
   # Relu     
    if OUTPUT_ACTIVATION == "relu":
        # acc = tl.maximum(acc, 1e-4)
        acc = tl.where(acc > THRESHOLD, acc, 0.0)

    if HAS_Z:
        out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=8),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_fwd_kernel_wip(
    # Pointers to matrices
    cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, B_ptr, prev_states_ptr, D_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_B_batch, stride_B_seqlen, stride_B_head, stride_B_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_n = tl.program_id(axis=0)
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    B_ptr += pid_b * stride_B_batch + pid_c * chunk_size * stride_B_seqlen + (pid_h // nheads_ngroups_ratio) * stride_B_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE)

    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
    B_ptrs = B_ptr + (offs_m[None, :] * stride_B_seqlen + offs_k_dstate[:, None] * stride_B_dstate)
    prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_m[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None, :] * stride_out_hdim)

    prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
    # if pid_c == 0:
    #     if pid_b == 0:
    #         if pid_h == 0:
    #             tl.device_print("", prev_states)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # scale_m = tl.exp(dA_cs_m)
    # C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
    # acc = tl.dot(C, prev_states.to(C_ptr.dtype.element_ty)) * scale_m[:, None]
    # cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_m[None, :] < chunk_size), other=0.0).to(tl.float32)
    # cb *= tl.exp((dA_cs_m[:, None] - dA_cs_m[None, :]))
    # dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # cb *= dt_m
    # mask = offs_m[:, None] >= offs_m[None, :]
    # cb = tl.where(mask, cb, 0.0)
    # cb = cb.to(x_ptr.dtype.element_ty)
    # x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0)
    # acc += tl.dot(cb, x)
    # if HAS_D:
    #     if D_HAS_HDIM:
    #         D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
    #     else:
    #         D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
    #     acc += x.to(tl.float32) * D
    # tl.store(out_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    for start_m in range(0, chunk_size_limit, BLOCK_SIZE_M):
        start_m = tl.multiple_of(start_m, BLOCK_SIZE_M)
        dA_cs_m = tl.load(dA_cumsum_ptr + (start_m + offs_m) * stride_dA_cs_csize, mask=offs_m < chunk_size - start_m, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_prev = tl.load(seq_idx_ptr + start_m - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
            seq_idx_m = tl.load(seq_idx_ptr + (start_m + offs_m) * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit - start_m, other=-1)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit - start_m) & (offs_k_dstate[None, :] < dstate), other=0.0)
        acc = tl.dot(C, prev_states.to(C_ptr.dtype.element_ty)) * scale_m[:, None]
        # cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size - start_m) & (offs_m[None, :] < chunk_size - start_m), other=0.0).to(tl.float32)
        # cb *= tl.exp((dA_cs_m[:, None] - dA_cs_m[None, :]))
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size - start_m, other=0.0).to(tl.float32)
        # cb *= dt_m
        # mask = offs_m[:, None] >= offs_m[None, :]
        # cb = tl.where(mask, cb, 0.0)
        # cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit - start_m) & (offs_n[None, :] < hdim), other=0.0)
        # acc += tl.dot(cb, x)

        if HAS_D:
            if D_HAS_HDIM:
                D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
            else:
                D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
            acc += x.to(tl.float32) * D

        # if HAS_Z:
        #     out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        #     out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
        #     tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        #     z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        #     z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        #     z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        #     acc *= z * tl.sigmoid(z)

        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit - start_m) & (offs_n[None, :] < hdim))

        # TODO: this is not correct, and quite a bit slower
        if start_m + BLOCK_SIZE_M < chunk_size_limit:
            # B = tl.load(B_ptrs, mask=(offs_m[None, :] < chunk_size_limit - start_m) & (offs_k_dstate[:, None] < dstate), other=0.0).to(tl.float32)
            B = tl.load(B_ptrs, mask=(offs_m[None, :] < chunk_size_limit - start_m) & (offs_k_dstate[:, None] < dstate), other=0.0)
            dA_cs_last = tl.load(dA_cumsum_ptr + (start_m + BLOCK_SIZE_M) * stride_dA_cs_csize).to(tl.float32)
            # TODO: seq_idx
            scale = tl.exp((dA_cs_last - dA_cs_m)) * dt_m
            # B *= scale
            B = B.to(x_ptr.dtype.element_ty)
            tmp = tl.dot(B, x)
            prev_states += tmp.to(prev_states.dtype)

        C_ptrs += BLOCK_SIZE_M * stride_C_seqlen
        B_ptrs += BLOCK_SIZE_M * stride_B_seqlen
        cb_ptrs += BLOCK_SIZE_M * stride_cb_csize_m + BLOCK_SIZE_M * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_M * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_M * stride_dt_csize
        out_ptrs += BLOCK_SIZE_M * stride_out_seqlen


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_M': 64}),
        triton.Config({'BLOCK_SIZE_M': 128}),
        triton.Config({'BLOCK_SIZE_M': 256}),
    ],
    key=["chunk_size", "hdim"],
)
@triton.jit
def _chunk_scan_bwd_dz_kernel(
    # Pointers to matrices
    dout_ptr, out_ptr, z_ptr, x_ptr, D_ptr, outz_ptr, dz_ptr, dout_x_ptr, dD_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_D_head,
    stride_outz_batch, stride_outz_seqlen, stride_outz_head, stride_outz_hdim,
    stride_dz_batch, stride_dz_seqlen, stride_dz_head, stride_dz_hdim,
    stride_doutx_batch, stride_doutx_seqlen, stride_doutx_head, stride_doutx_hdim,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_DDACS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    output_activation: tl.constexpr,
    threshold: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dout_x_ptr += pid_b * stride_doutx_batch + pid_c * chunk_size * stride_doutx_seqlen + pid_h * stride_doutx_head
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
    dz_ptr += pid_b * stride_dz_batch + pid_c * chunk_size * stride_dz_seqlen + pid_h * stride_dz_head
    if RECOMPUTE_OUTPUT:
        outz_ptr += pid_b * stride_outz_batch + pid_c * chunk_size * stride_outz_seqlen + pid_h * stride_outz_head
    if HAS_DDACS:
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    if HAS_D:
        x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dout_x_ptrs = dout_x_ptr + (offs_m[:, None] * stride_doutx_seqlen + offs_n[None, :] * stride_doutx_hdim)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None, :] * stride_out_hdim)
    z_ptrs = z_ptr + (offs_m[:, None] * stride_z_seqlen + offs_n[None, :] * stride_z_hdim)
    dz_ptrs = dz_ptr + (offs_m[:, None] * stride_dz_seqlen + offs_n[None, :] * stride_dz_hdim)
    if RECOMPUTE_OUTPUT:
        outz_ptrs = outz_ptr + (offs_m[:, None] * stride_outz_seqlen + offs_n[None, :] * stride_outz_hdim)
    if HAS_D:
        x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    out = tl.load(out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    z = tl.load(z_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    z_sigmoid = tl.sigmoid(z)
    # modify it here for relu
    if output_activation == "relu":
        relu_mask = out > threshold
    if RECOMPUTE_OUTPUT:
        outz = out * z * z_sigmoid
        tl.store(outz_ptrs, outz, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    dz = dout * out * z_sigmoid * (1 + z * (1 - z_sigmoid))
    tl.store(dz_ptrs, dz, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    dout *= z * z_sigmoid
    dout = tl.where(relu_mask, dout, 0.0)
    tl.store(dout_x_ptrs, dout, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    if HAS_D:
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            dD = tl.sum(dout * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            dD = tl.sum(dout * x)
            tl.store(dD_ptr, dD)
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        out -= x * D
    if HAS_DDACS:
        ddA_cs = tl.sum(dout * out, axis=1)
        tl.store(ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size)


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
def _chunk_scan_bwd_dstates_kernel(
    # Pointers to matrices
    dout_ptr, c_ptr, dprev_states_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nchunks, nheads_ngroups_ratio,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_c_batch, stride_c_seqlen, stride_c_head, stride_c_dstate,
    stride_dprev_states_batch, stride_dprev_states_chunk, stride_dprev_states_head, stride_dprev_states_hdim, stride_dprev_states_dstate,
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
    c_ptr += pid_b * stride_c_batch + pid_c * chunk_size * stride_c_seqlen + (pid_h // nheads_ngroups_ratio) * stride_c_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_hdim + offs_k[None, :] * stride_dout_seqlen)
    c_ptrs = c_ptr + (offs_n[None, :] * stride_c_dstate + offs_k[:, None] * stride_c_seqlen)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale_k = tl.exp(dA_cs_k)
        else:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
            scale_k = tl.where(seq_idx_k == seq_idx_prev, tl.exp(dA_cs_k), 0.0)
        dout = (dout * scale_k).to(dout_ptr.dtype.element_ty)
        c = tl.load(c_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0)
        acc += tl.dot(dout, c)
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        c_ptrs += BLOCK_SIZE_K * stride_c_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    out = acc.to(dprev_states_ptr.dtype.element_ty)

    dprev_states_ptr += pid_b * stride_dprev_states_batch + pid_c * stride_dprev_states_chunk + pid_h * stride_dprev_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dprev_states_ptrs = dprev_states_ptr + (offs_m[:, None] * stride_dprev_states_hdim + offs_n[None, :] * stride_dprev_states_dstate)
    tl.store(dprev_states_ptrs, out, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate))


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
def _chunk_scan_bwd_dc_kernel(
    # Pointers to matrices
    dout_ptr, prev_states_ptr, C_ptr, dA_cumsum_ptr, seq_idx_ptr,
    dc_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, dstate, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_prev_states_batch, stride_prev_states_chunk, stride_prev_states_head, stride_prev_states_hdim, stride_prev_states_dstate,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dc_batch, stride_dc_seqlen, stride_dc_split, stride_dc_group, stride_dc_dstate,
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
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dout_head
    dc_ptr += pid_b * stride_dc_batch + pid_c * chunk_size * stride_dc_seqlen + pid_g * stride_dc_group + pid_s * stride_dc_split
    prev_states_ptr += pid_b * stride_prev_states_batch + pid_c * stride_prev_states_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_prev_states_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dA_cs_head
    if HAS_DDA_CS:
        C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + pid_g * stride_C_head
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_ddA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_prev_states_dstate + offs_k[:, None] * stride_prev_states_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    if HAS_DDA_CS:
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_n[None, :] * stride_C_dstate)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        c = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    for h in range(nheads_iter):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
        prev_states = tl.load(prev_states_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0)
        prev_states = prev_states.to(dout_ptrs.dtype.element_ty)
        dc = tl.dot(dout, prev_states)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_m)
        else:
            scale = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        dc *= scale[:, None]
        if HAS_DDA_CS:
            ddA_cs = tl.sum(dc * c, axis=1)
            tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
        acc += dc
        dout_ptrs += stride_dout_head
        prev_states_ptrs += stride_prev_states_head
        dA_cumsum_ptrs += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head
    # if HAS_SEQ_IDX:
    #     seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
    #     seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    #     acc = tl.where(seq_idx_m[:, None] == seq_idx_prev, acc, 0.0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dc_ptrs = dc_ptr + (offs_m[:, None] * stride_dc_seqlen + offs_n[None, :] * stride_dc_dstate)
    tl.store(dc_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))


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
    key=['chunk_size', 'hdim'],
)
@triton.jit
def _chunk_scan_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, cb_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, D_ptr,
    dx_ptr, ddt_ptr, # dD_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_D_head,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    # stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_hdim, stride_dD_csize,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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
    # if HAS_D:
    #     dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Idk why limiting K_MAX gives wrong results, is it a Triton bug?
    # K_MAX = min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    K_MAX = chunk_size_limit
    for k in range(0, K_MAX, BLOCK_SIZE_K):
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
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)

    # if HAS_D:
    #     dout_new_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize + offs_n[None, :] * stride_dout_hdim)
    #     dout = tl.load(dout_new_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
    #     dD = tl.sum(x * dout, axis=0)
    #     tl.store(dD_ptr + offs_n * stride_dD_hdim, dD, mask=offs_n < N)


# Disabling HAS_DDA_CS for now since it's much slower
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 16}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 16}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 128}, num_stages=4, num_warps=8),
    ],
    key=['chunk_size', 'hdim'],
)
# @triton.heuristics({"BLOCK_SIZE_N": lambda args: max(triton.next_power_of_2(args["chunk_size"]), 16)})
# @triton.heuristics({"BLOCK_SIZE_N": lambda args: 32})
@triton.jit
def _chunk_scan_bwd_dcb_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, cb_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    dcb_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dcb_batch, stride_dcb_chunk, stride_dcb_split, stride_dcb_group, stride_dcb_csize_m, stride_dcb_csize_n,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize_m, stride_ddA_cs_csize_n,
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
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_x_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dA_cs_head
    if HAS_DDA_CS:
        cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_g * stride_cb_head
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_ddA_cs_head + pid_m * stride_ddA_cs_csize_m
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    if HAS_DDA_CS:
        cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_n * stride_ddA_cs_csize_n

    if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
        dcb_ptr += pid_b * stride_dcb_batch + pid_c * stride_dcb_chunk + pid_g * stride_dcb_group + pid_s * stride_dcb_split
        dcb_ptrs = dcb_ptr + (offs_m[:, None] * stride_dcb_csize_m + offs_n[None, :] * stride_dcb_csize_n)
        tl.store(dcb_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dcb_ptr.dtype.element_ty), mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
        return

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    chunk_size_limit_n = min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size), other=0.0).to(tl.float32)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    for h in range(nheads_iter):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n), other=0.0)
        dcb = tl.dot(dout, x)
        dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size, other=0.0).to(tl.float32)
        dcb *= dt_n
        dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        dA_cs_n = tl.load(dA_cumsum_ptr + offs_n * stride_dA_cs_csize, mask=offs_n < chunk_size_limit, other=0.0).to(tl.float32)
        # dcb *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
        dcb *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_n[None, :]), 0.0))
        if HAS_DDA_CS:
            tl.static_assert(not HAS_SEQ_IDX, "HAS_SEQ_IDX not supported with HAS_DDA_CS yet")
            ddA_cs = dcb * cb
            mask = offs_m[:, None] >= offs_n[None, :] + 1
            ddA_cs = tl.where(mask, ddA_cs, 0.0)
            ddA_cs = tl.cumsum(ddA_cs, axis=1)
            ddA_cs = tl.where(mask, ddA_cs, 0.0)
            ddA_cs = tl.sum(ddA_cs, axis=0)
            tl.store(ddA_cumsum_ptrs + stride_ddA_cs_csize_n, ddA_cs, mask=offs_n < chunk_size - 1)
            tl.store(ddA_cumsum_ptr, 0.0)
        acc += dcb
        dout_ptrs += stride_dout_head
        x_ptrs += stride_x_head
        dt_ptrs += stride_dt_head
        dA_cumsum_ptr += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptr += stride_ddA_cs_head
            ddA_cumsum_ptrs += stride_ddA_cs_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    mask = offs_m[:, None] >= offs_n[None, :]
    acc = tl.where(mask, acc, 0.0)
    dcb_ptr += pid_b * stride_dcb_batch + pid_c * stride_dcb_chunk + pid_g * stride_dcb_group + pid_s * stride_dcb_split
    dcb_ptrs = dcb_ptr + (offs_m[:, None] * stride_dcb_csize_m + offs_n[None, :] * stride_dcb_csize_n)
    tl.store(dcb_ptrs, acc, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


# Not numerically stable and should not be used. Leaving here for reference.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_M': 64}),
        triton.Config({'BLOCK_SIZE_M': 128}),
        triton.Config({'BLOCK_SIZE_M': 256}),
    ],
    key=["chunk_size", "hdim"],
)
@triton.jit
def _chunk_scan_bwd_ddAcs_unstable_kernel(
    # Pointers to matrices
    dout_ptr, out_ptr, dt_ptr, ddt_ptr, x_ptr, D_ptr,
    ddA_cumsum_ptr, dD_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_D_head,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    SUBTRACT_DDTDT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    if HAS_D:
        x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None, :] * stride_out_hdim)
    if HAS_D:
        x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    out = tl.load(out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            dD = tl.sum(dout * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            dD = tl.sum(dout * x)
            tl.store(dD_ptr, dD)
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        out -= x * D
    ddA_cs = tl.sum(dout * out, axis=1)
    if SUBTRACT_DDTDT:
        dt = tl.load(dt_ptr + offs_m * stride_dt_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        ddt = tl.load(ddt_ptr + offs_m * stride_ddt_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        ddA_cs -= dt * ddt
    tl.store(ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128}, num_stages=4, num_warps=8),
    ],
    key=['chunk_size', 'hdim'],
)
@triton.jit
def _chunk_scan_bwd_ddAcs_stable_kernel_old(
    # Pointers to matrices
    x_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, cb_ptr,
    ddAcs_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    stride_ddAcs_batch, stride_ddAcs_chunk, stride_ddAcs_head, stride_ddAcs_csize_m, stride_ddAcs_csize_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    chunk_size_limit_n = min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    # Doing a matmul loop with cumsum later on will cause Triton to crash
    # Instead we do just one big matmul
    # acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # for k in range(0, hdim, BLOCK_SIZE_K):
    #     dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim - k), other=0.0)
    #     x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim - k) & (offs_n[None, :] < chunk_size_limit), other=0.0)
    #     acc += tl.dot(dout, x)
    #     dout_ptrs += BLOCK_SIZE_K * stride_dout_hdim
    #     x_ptrs += BLOCK_SIZE_K * stride_x_hdim
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
    x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n), other=0.0)
    acc = tl.dot(dout, x)
    cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size), other=0.0).to(tl.float32)
    acc *= cb
    dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size, other=0.0).to(tl.float32)
    acc *= dt_n
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    dA_cs_n = tl.load(dA_cumsum_ptr + offs_n * stride_dA_cs_csize, mask=offs_n < chunk_size, other=0.0).to(tl.float32)
    # acc *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
    acc *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_n[None, :]), 0.0))
    mask = offs_m[:, None] >= offs_n[None, :] + 1
    acc = tl.where(mask, acc, 0.0)
    acc = tl.cumsum(acc, axis=1)
    acc = tl.where(mask, acc, 0.0)
    ddA_cs = tl.sum(acc, axis=0)
    ddAcs_ptr += pid_b * stride_ddAcs_batch + pid_c * stride_ddAcs_chunk + pid_h * stride_ddAcs_head + pid_m * stride_ddAcs_csize_m
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ddAcs_ptrs = ddAcs_ptr + offs_n * stride_ddAcs_csize_n
    tl.store(ddAcs_ptrs + stride_ddAcs_csize_n, ddA_cs, mask=offs_n < chunk_size - 1)
    tl.store(ddAcs_ptr, 0.0)

    # offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, 64)
    # offs_k = tl.arange(0, BLOCK_SIZE_K)
    # dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    # x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    # dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    # cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)

    # chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    # chunk_size_limit_n = min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    # rowsum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    # dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
    # dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # ddAcs_ptr += pid_b * stride_ddAcs_batch + pid_c * stride_ddAcs_chunk + pid_h * stride_ddAcs_head + pid_m * stride_ddAcs_csize_m
    # ddAcs_ptrs = ddAcs_ptr + offs_n * stride_ddAcs_csize_n
    # for n in range(0, chunk_size_limit_n, 64):
    #     x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n - n), other=0.0)
    #     acc = tl.dot(dout, x)
    #     cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size - n), other=0.0).to(tl.float32)
    #     acc *= cb
    #     dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size - n, other=0.0).to(tl.float32)
    #     acc *= dt_n
    #     dA_cs_n = tl.load(dA_cumsum_ptr + offs_n * stride_dA_cs_csize, mask=offs_n < chunk_size - n, other=0.0).to(tl.float32)
    #     acc *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
    #     mask = offs_m[:, None] >= offs_n[None, :] + 1 + n
    #     acc = tl.where(mask, acc, 0.0)
    #     acc = tl.cumsum(acc, axis=1)
    #     acc = tl.where(mask, acc, 0.0)
    #     ddA_cs = tl.sum(acc, axis=0)
    #     tl.store(ddAcs_ptrs, ddA_cs, mask=offs_n < chunk_size - 1 - n)
    # # tl.store(ddAcs_ptr, 0.0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
    ],
    key=['chunk_size', 'hdim'],
)
@triton.jit
def _chunk_scan_bwd_ddAcs_stable_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, cb_ptr,
    ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize_m, stride_ddA_cs_csize_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head + pid_m * stride_ddA_cs_csize_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)
    ddAcs_ptrs = ddA_cumsum_ptr + offs_n * stride_ddA_cs_csize_n
    tl.store(ddA_cumsum_ptr, 0.0)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    rowsum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # Actually hi is (pid_m + 1) * BLOCK_SIZE_M - 1 but subtracting 1 makes it slower
    lo, hi = 0, (pid_m + 1) * BLOCK_SIZE_M
    # lo, hi = 0, chunk_size
    for start_n in range(lo, hi, BLOCK_SIZE_N):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE_N)
        # Doing a matmul loop with cumsum later on will cause Triton to crash
        # Instead we do just one big matmul
        # acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # for k in range(0, hdim, BLOCK_SIZE_K):
        #     dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim - k), other=0.0)
        #     x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim - k) & (offs_n[None, :] < chunk_size_limit), other=0.0)
        #     acc += tl.dot(dout, x)
        #     dout_ptrs += BLOCK_SIZE_K * stride_dout_hdim
        #     x_ptrs += BLOCK_SIZE_K * stride_x_hdim
        # x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n), other=0.0)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit - start_n), other=0.0)
        acc = tl.dot(dout, x)
        dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size - start_n, other=0.0).to(tl.float32)
        acc *= dt_n
        # If there's seq_idx, we already zero'ed out cb[i, j] for seq_idx[i] != seq_idx[j]
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size - start_n), other=0.0).to(tl.float32)
        acc *= cb
        dA_cs_n = tl.load(dA_cumsum_ptr + (start_n + offs_n) * stride_dA_cs_csize, mask=offs_n < chunk_size - start_n, other=0.0).to(tl.float32)
        # acc *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
        acc *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_n[None, :]), 0.0))
        mask = offs_m[:, None] >= start_n + offs_n[None, :] + 1
        acc = tl.where(mask, acc, 0.0)
        rowsum_new = rowsum + tl.sum(acc, axis=1)
        acc = rowsum[:, None] + tl.cumsum(acc, axis=1)
        rowsum = rowsum_new
        acc = tl.where(mask, acc, 0.0)
        ddA_cs = tl.sum(acc, axis=0)
        tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, ddA_cs, mask=offs_n < chunk_size - start_n - 1)
        x_ptrs += BLOCK_SIZE_N * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_N * stride_dt_csize
        cb_ptrs += BLOCK_SIZE_N * stride_cb_csize_n
        ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n

    # Need to zero out the rest, since we'll be summing the rows together
    for start_n in range(hi, chunk_size, BLOCK_SIZE_N):
        tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32), mask=offs_n < chunk_size - start_n - 1)
        ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'dstate', 'hdim'],
)
@triton.jit
def _chunk_scan_bwd_ddAcs_prev_kernel(
    # Pointers to matrices
    dout_ptr, prev_states_ptr, C_ptr, dA_cumsum_ptr, seq_idx_ptr,
    ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, dstate, hdim,
    batch, seqlen, nchunks, nheads_ngroups_ratio,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_prev_states_batch, stride_prev_states_chunk, stride_prev_states_head, stride_prev_states_hdim, stride_prev_states_dstate,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
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
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    prev_states_ptr += pid_b * stride_prev_states_batch + pid_c * stride_prev_states_chunk + pid_h * stride_prev_states_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_prev_states_dstate + offs_k[:, None] * stride_prev_states_hdim)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_n[None, :] * stride_C_dstate)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
    prev_states = tl.load(prev_states_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0)
    prev_states = prev_states.to(dout_ptrs.dtype.element_ty)
    acc = tl.dot(dout, prev_states)
    c = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    ddA_cs = tl.sum(acc * c, axis=1)
    dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    if not HAS_SEQ_IDX:
        scale = tl.exp(dA_cs_m)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        scale =  tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
    ddA_cs *= scale
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)


def _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D=None, z=None, seq_idx=None, output_activation=None, threshold=0.0):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # Allocates output.
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
    if z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3))
                  if z is not None else (0, 0, 0, 0))
    _chunk_scan_fwd_kernel[grid](
        cb, x, z, out, out_x, dt, dA_cumsum, seq_idx, C, states, D,
        chunk_size, headdim, dstate,
        batch, seqlen, nheads // ngroups,
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
        D.stride(0) if D is not None else 0,
        threshold,
        True,
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
        OUTPUT_ACTIVATION=output_activation,
    )
    return out, out_x


def _chunk_scan_fwd_wip(cb, x, dt, dA_cumsum, C, B, states, D=None, z=None, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert B.shape == C.shape
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # Allocates output.
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
    if z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3))
                  if z is not None else (0, 0, 0, 0))
    _chunk_scan_fwd_kernel_wip[grid](
        cb, x, z, out, out_x, dt, dA_cumsum, seq_idx, C, B, states, D,
        chunk_size, headdim, dstate,
        batch, seqlen, nheads // ngroups,
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
        D.stride(0) if D is not None else 0,
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        BLOCK_SIZE_M=128,
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
    )
    return out, out_x


def _chunk_scan_bwd_dz(x, z, out, dout, chunk_size, has_ddAcs=True, D=None, dz=None, recompute_output=False):
    batch, seqlen, nheads, headdim = x.shape
    assert z.shape == x.shape
    assert out.shape == x.shape
    assert dout.shape == out.shape
    nchunks = math.ceil(seqlen / chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
    if has_ddAcs:
        ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=x.device, dtype=torch.float32)
    if D is not None:
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    if dz is not None:
        assert dz.shape == z.shape
    else:
        dz = torch.empty_like(z)
    if recompute_output:
        outz = torch.empty_like(x)
    dout_x = torch.empty_like(dout)
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    grid_dz = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dz_kernel[grid_dz](
            dout, out, z, x, D, outz if recompute_output else None,
            dz, dout_x, dD, ddA_cumsum if has_ddAcs else None,
            chunk_size, headdim,
            batch, seqlen,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            D.stride(0) if D is not None else 0,
            *((outz.stride(0), outz.stride(1), outz.stride(2), outz.stride(3)) if recompute_output else (0, 0, 0, 0)),
            dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3),
            dout_x.stride(0), dout_x.stride(1), dout_x.stride(2), dout_x.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            *((ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3))
              if has_ddAcs else (0, 0, 0, 0)),
            D is not None,
            D.dim() == 2 if D is not None else True,
            has_ddAcs,
            BLOCK_SIZE_N=max(triton.next_power_of_2(headdim), 16),
            RECOMPUTE_OUTPUT=recompute_output,
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_bwd_dz_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return_vals = (dz, dout_x, dD, ddA_cumsum) if has_ddAcs else (dz, dout_x, dD)
    return return_vals if not recompute_output else (*return_vals, outz)


def _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=None, dtype=None):
    batch, seqlen, nheads, headdim = dout.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    dtype = C.dtype if dtype is None else dtype
    dprev_states = torch.empty(batch, nchunks, nheads, headdim, dstate, device=C.device, dtype=dtype)
    grid_dstates = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                            batch * nchunks, nheads)
    with torch.cuda.device(C.device.index):
        _chunk_scan_bwd_dstates_kernel[grid_dstates](
            dout, C, dprev_states, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nchunks, nheads // ngroups,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            dprev_states.stride(0), dprev_states.stride(1), dprev_states.stride(2), dprev_states.stride(3), dprev_states.stride(4),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return dprev_states


def _chunk_scan_bwd_dC(prev_states, dA_cumsum, dout, seq_idx=None, C=None, ngroups=1):
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == (batch, seqlen, nheads, headdim)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if C is not None:
        assert C.shape == (batch, seqlen, ngroups, dstate)
        C_strides = (C.stride(0), C.stride(1), C.stride(2), C.stride(3))
        ddA_cumsum_prev = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
        ddA_cumsum_prev_strides = (ddA_cumsum_prev.stride(0), ddA_cumsum_prev.stride(2), ddA_cumsum_prev.stride(1), ddA_cumsum_prev.stride(3))
    else:
        C_strides = (0, 0, 0, 0)
        ddA_cumsum_prev = None
        ddA_cumsum_prev_strides = (0, 0, 0, 0)
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(dout.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dC = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=dout.device, dtype=torch.float32)
    grid_dc = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                        batch * nchunks, nsplits * ngroups)
    with torch.cuda.device(dout.device.index):
        _chunk_scan_bwd_dc_kernel[grid_dc](
            dout, prev_states, C, dA_cumsum, seq_idx, dC, ddA_cumsum_prev,
            chunk_size, dstate, headdim,
            batch, seqlen, nheads, nheads_per_program, ngroups,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3), prev_states.stride(4),
            *C_strides,
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dC.stride(0), dC.stride(1), dC.stride(2), dC.stride(3), dC.stride(4),
            *ddA_cumsum_prev_strides,
            HAS_DDA_CS=ddA_cumsum_prev is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    dC = dC.sum(2)
    return dC if C is None else (dC, ddA_cumsum_prev)


def _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=None, CB=None, ngroups=1):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if CB is not None:
        assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
        CB_strides = (CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4))
        BLOCK_SIZE_M_min = 16
        ddA_cumsum = torch.empty(batch, nheads, nchunks, triton.cdiv(chunk_size, BLOCK_SIZE_M_min),
                                chunk_size, device=x.device, dtype=torch.float32)
        ddA_cumsum_strides = (ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4))
    else:
        CB_strides = (0, 0, 0, 0, 0)
        ddA_cumsum = None
        ddA_cumsum_strides = (0, 0, 0, 0, 0)
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dcb = torch.empty(batch, nchunks, nsplits, ngroups, chunk_size, chunk_size, device=x.device, dtype=torch.float32)
    grid_dcb = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']),
                        batch * nchunks, nsplits * ngroups)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dcb_kernel[grid_dcb](
            x, dout, CB, dt, dA_cumsum, seq_idx, dcb, ddA_cumsum,
            chunk_size, headdim,
            batch, seqlen, nheads, nheads_per_program, ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            *CB_strides,
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dcb.stride(0), dcb.stride(1), dcb.stride(2), dcb.stride(3), dcb.stride(4), dcb.stride(5),
            *ddA_cumsum_strides,
            HAS_DDA_CS=ddA_cumsum is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    dcb = dcb.sum(2)
    if ddA_cumsum is not None:
        BLOCK_SIZE_M_actual = _chunk_scan_bwd_dcb_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1) // BLOCK_SIZE_M_actual
        ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return dcb if CB is None else (dcb, ddA_cumsum)


def _chunk_scan_bwd_dx(cb, x, dt, dA_cumsum, dout, D=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    ngroups = cb.shape[2]
    assert nheads % ngroups == 0
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    # if D is not None:
    #     BLOCK_SIZE_M_min = 32
    #     dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_M_min), batch, nchunks, nheads, headdim, device=D.device, dtype=torch.float32)
    # else:
    #     dD = None
    dx = torch.empty_like(x)
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                        batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dx_kernel[grid_dx](
            x, cb, dout, dt, dA_cumsum, D, dx, ddt, # dD,
            chunk_size, headdim,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(-1), cb.stride(-2),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            D.stride(0) if D is not None else 0,
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            # dD.stride(1) if dD is not None else 0, dD.stride(2) if dD is not None else 0, dD.stride(3) if dD is not None else 0, dD.stride(4) if dD is not None else 0, dD.stride(0) if dD is not None else 0,
            D is not None,
            D.dim() == 2 if D is not None else True,
        )
    # if D is not None:
    #     BLOCK_SIZE_actual = _chunk_scan_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
    #     n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    #     dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
    return dx, ddt.to(dtype=dt.dtype)


def _chunk_scan_bwd_ddAcs_unstable(x, dt, out, dout, ddt, D=None, subtract_ddtdt=True):
    """Not numerically stable and should not be used. Leaving here for reference.
    """

    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert ddt.shape == dt.shape
    assert out.shape == x.shape
    assert dout.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    ddA_cumsum = torch.empty_like(dt)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    if D is not None:  # Triton gives wrong results if we write to the same location
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_ddAcs_unstable_kernel[grid_ddtcs](
            dout, out, dt, ddt, x, D, ddA_cumsum, dD,
            chunk_size, headdim,
            batch, seqlen,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            D.stride(0) if D is not None else 0,
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            subtract_ddtdt,
            BLOCK_SIZE_N=max(triton.next_power_of_2(headdim), 16),
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_bwd_ddAcs_unstable_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return ddA_cumsum, dD


def _chunk_scan_bwd_ddAcs_stable_old(x, dt, dA_cumsum, dout, cb):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == x.shape
    assert dA_cumsum.shape == dt.shape
    ngroups = cb.shape[2]
    assert nheads % ngroups == 0
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    BLOCK_SIZE_M_min = 16
    ddA_cumsum = torch.empty(batch, nheads, nchunks, triton.cdiv(chunk_size, BLOCK_SIZE_M_min),
                             chunk_size, device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_ddAcs_stable_kernel_old[grid_ddtcs](
            x, dout, dt, dA_cumsum, cb, ddA_cumsum,
            chunk_size, headdim,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4),
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
            BLOCK_SIZE_N=max(triton.next_power_of_2(chunk_size), 16),
        )
    BLOCK_SIZE_M_actual = _chunk_scan_bwd_ddAcs_stable_kernel_old.best_config.kwargs["BLOCK_SIZE_M"]
    n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1) // BLOCK_SIZE_M_actual
    ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return ddA_cumsum


def _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, cb):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == x.shape
    assert dA_cumsum.shape == dt.shape
    ngroups = cb.shape[2]
    assert nheads % ngroups == 0
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    BLOCK_SIZE_M_min = 32
    ddA_cumsum = torch.empty(batch, nheads, nchunks, triton.cdiv(chunk_size, BLOCK_SIZE_M_min),
                             chunk_size, device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_ddAcs_stable_kernel[grid_ddtcs](
            x, dout, dt, dA_cumsum, cb, ddA_cumsum,
            chunk_size, headdim,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4),
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    BLOCK_SIZE_M_actual = _chunk_scan_bwd_ddAcs_stable_kernel.best_config.kwargs["BLOCK_SIZE_M"]
    n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1) // BLOCK_SIZE_M_actual
    ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return ddA_cumsum


def _chunk_scan_bwd_ddAcs_prev(prev_states, C, dout, dA_cumsum, seq_idx=None):
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == (batch, seqlen, nheads, headdim)
    ngroups = C.shape[2]
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    ddA_cumsum_prev = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    grid_ddAcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                          batch * nchunks, nheads)
    with torch.cuda.device(dout.device.index):
        _chunk_scan_bwd_ddAcs_prev_kernel[grid_ddAcs](
            dout, prev_states, C, dA_cumsum, seq_idx, ddA_cumsum_prev,
            chunk_size, dstate, headdim,
            batch, seqlen, nchunks, nheads // ngroups,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3), prev_states.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            ddA_cumsum_prev.stride(0), ddA_cumsum_prev.stride(2), ddA_cumsum_prev.stride(1), ddA_cumsum_prev.stride(3),
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    return ddA_cumsum_prev

# =======================================================================================================


#dr
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
    ],
    key=['chunk_size'],
)
@triton.jit
def _chunk_scan_bwd_ddAcs_stable_dr_kernel(
    # Inputs - using precomputed x_masked and M
    x_masked_ptr, M_ptr, dout_ptr, CB_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Output
    ddA_ptr,
    # Dimensions
    chunk_size, hdim, batch, seqlen, ngroups,
    # Strides x_masked (b, l, h, p)
    stride_xm_batch, stride_xm_seqlen, stride_xm_head, stride_xm_hdim,
    # Strides M (b, l, h, p)
    stride_M_batch, stride_M_seqlen, stride_M_head, stride_M_hdim,
    # Strides dout (b, l, h, p)
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    # Strides CB (b, c, g, l, l) - CB is still per group!
    stride_CB_batch, stride_CB_chunk, stride_CB_group, stride_CB_csize_m, stride_CB_csize_n,
    # Strides dA_cumsum (b, h, c, l, p)
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
    # Strides output ddA (b, l, h, p)
    stride_ddA_batch, stride_ddA_seqlen, stride_ddA_head, stride_ddA_hdim,
    # Seq idx
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)   # head index (nheads)
    pid_t = tl.program_id(axis=0)   # time position
    
    # For CB: need group index
    nheads_per_group = stride_dout_head // stride_CB_group if stride_CB_group > 0 else 1
    
    if pid_t >= chunk_size:
        return
        
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    if pid_t >= chunk_size_limit:
        return
    
    # Offsets for x_masked, M, dout, dA_cumsum - all use pid_h directly
    x_masked_ptr += pid_b * stride_xm_batch + pid_c * chunk_size * stride_xm_seqlen + pid_h * stride_xm_head
    M_ptr += pid_b * stride_M_batch + pid_c * chunk_size * stride_M_seqlen + pid_h * stride_M_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * chunk_size * stride_ddA_seqlen + pid_h * stride_ddA_head
    
    # CB uses group_id
    nheads_ngroups_ratio = stride_dout_head // stride_CB_group  # heads per group
    group_id = pid_h // nheads_ngroups_ratio
    CB_ptr += pid_b * stride_CB_batch + pid_c * stride_CB_chunk + group_id * stride_CB_group
    
    offs_p = tl.arange(0, BLOCK_SIZE)
    mask_p = offs_p < hdim
    
    # Load S_t
    dA_t = tl.load(dA_cumsum_ptr + pid_t * stride_dA_cs_csize + offs_p * stride_dA_cs_hdim, mask=mask_p).to(tl.float32)
    exp_S_t = tl.exp(dA_t)
    exp_neg_S_t = tl.exp(-dA_t)
    
    # Load precomputed x_masked_t and dout_t
    dout_t = tl.load(dout_ptr + pid_t * stride_dout_seqlen + offs_p * stride_dout_hdim, mask=mask_p).to(tl.float32)
    x_masked_t = tl.load(x_masked_ptr + pid_t * stride_xm_seqlen + offs_p * stride_xm_hdim, mask=mask_p).to(tl.float32)
    
    pos_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    neg_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop j = 0 to t (positive term)
    for j in range(pid_t + 1):
        # Load precomputed x_masked_j
        x_masked_j = tl.load(x_masked_ptr + j * stride_xm_seqlen + offs_p * stride_xm_hdim, mask=mask_p).to(tl.float32)
        
        # CB is per-group and is a scalar per (i,j) position, not per headdim
        CB_tj = tl.load(CB_ptr + pid_t * stride_CB_csize_m + j * stride_CB_csize_n).to(tl.float32)
        
        dA_j = tl.load(dA_cumsum_ptr + j * stride_dA_cs_csize + offs_p * stride_dA_cs_hdim, mask=mask_p).to(tl.float32)
        
        L_tj = dout_t * x_masked_j * CB_tj
        pos_acc += L_tj * tl.exp(-dA_j)
    
    # Loop i = t to end (negative term)
    for i in range(pid_t, chunk_size_limit):
        dout_i = tl.load(dout_ptr + i * stride_dout_seqlen + offs_p * stride_dout_hdim, mask=mask_p).to(tl.float32)
        # CB is per-group and is a scalar per (i,j) position, not per headdim
        CB_it = tl.load(CB_ptr + i * stride_CB_csize_m + pid_t * stride_CB_csize_n).to(tl.float32)
        dA_i = tl.load(dA_cumsum_ptr + i * stride_dA_cs_csize + offs_p * stride_dA_cs_hdim, mask=mask_p).to(tl.float32)
        
        L_it = dout_i * x_masked_t * CB_it
        neg_acc += L_it * tl.exp(dA_i)
    
    ddA_t = exp_S_t * pos_acc - exp_neg_S_t * neg_acc
    tl.store(ddA_ptr + pid_t * stride_ddA_seqlen + offs_p * stride_ddA_hdim, ddA_t.to(ddA_ptr.dtype.element_ty), mask=mask_p)


#dr
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_M': 64}),
        triton.Config({'BLOCK_SIZE_M': 128}),
        triton.Config({'BLOCK_SIZE_M': 256}),
    ],
    key=["chunk_size", "hdim"],
)
@triton.jit
def _chunk_scan_bwd_dr_dz_kernel(
    # Pointers to matrices
    dout_ptr, out_ptr, z_ptr, x_ptr, D_ptr, outz_ptr, dz_ptr, dout_x_ptr, dD_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_D_head,
    stride_outz_batch, stride_outz_seqlen, stride_outz_head, stride_outz_hdim,
    stride_dz_batch, stride_dz_seqlen, stride_dz_head, stride_dz_hdim,
    stride_doutx_batch, stride_doutx_seqlen, stride_doutx_head, stride_doutx_hdim,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_DDACS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    output_activation: tl.constexpr,
    threshold: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dout_x_ptr += pid_b * stride_doutx_batch + pid_c * chunk_size * stride_doutx_seqlen + pid_h * stride_doutx_head
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
    dz_ptr += pid_b * stride_dz_batch + pid_c * chunk_size * stride_dz_seqlen + pid_h * stride_dz_head
    if RECOMPUTE_OUTPUT:
        outz_ptr += pid_b * stride_outz_batch + pid_c * chunk_size * stride_outz_seqlen + pid_h * stride_outz_head
    if HAS_DDACS:
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    if HAS_D:
        x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dout_x_ptrs = dout_x_ptr + (offs_m[:, None] * stride_doutx_seqlen + offs_n[None, :] * stride_doutx_hdim)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None, :] * stride_out_hdim)
    z_ptrs = z_ptr + (offs_m[:, None] * stride_z_seqlen + offs_n[None, :] * stride_z_hdim)
    dz_ptrs = dz_ptr + (offs_m[:, None] * stride_dz_seqlen + offs_n[None, :] * stride_dz_hdim)
    if RECOMPUTE_OUTPUT:
        outz_ptrs = outz_ptr + (offs_m[:, None] * stride_outz_seqlen + offs_n[None, :] * stride_outz_hdim)
    if HAS_D:
        x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    out = tl.load(out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    z = tl.load(z_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    z_sigmoid = tl.sigmoid(z)
    # modify it here for relu
    if output_activation == "relu":
        relu_mask = out > threshold
    if RECOMPUTE_OUTPUT:
        outz = out * z * z_sigmoid
        tl.store(outz_ptrs, outz, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    dz = dout * out * z_sigmoid * (1 + z * (1 - z_sigmoid))
    tl.store(dz_ptrs, dz, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    dout *= z * z_sigmoid
    dout = tl.where(relu_mask, dout, 0.0)
    tl.store(dout_x_ptrs, dout, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    if HAS_D:
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            dD = tl.sum(dout * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            dD = tl.sum(dout * x)
            tl.store(dD_ptr, dD)
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        out -= x * D
    if HAS_DDACS:
        ddA_cs = tl.sum(dout * out, axis=1)
        tl.store(ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size)

#dr
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
def _chunk_scan_bwd_dr_dstates_kernel(
    # Pointers to matrices
    dout_ptr, c_ptr, dprev_states_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nchunks, nheads_ngroups_ratio,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_c_batch, stride_c_seqlen, stride_c_head, stride_c_dstate,
    stride_dprev_states_batch, stride_dprev_states_chunk, stride_dprev_states_head, stride_dprev_states_hdim, stride_dprev_states_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
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
    c_ptr += pid_b * stride_c_batch + pid_c * chunk_size * stride_c_seqlen + (pid_h // nheads_ngroups_ratio) * stride_c_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    # E_ptr += pid_b * stride_E_batch + pid_c * chunk_size * stride_E_seqlen + pid_h * stride_E_head
    # I_ptr += pid_b * stride_I_batch + pid_c * chunk_size * stride_I_seqlen + pid_h * stride_I_head
    # E_mean_ptr += pid_b * stride_Emean_batch + pid_c * chunk_size * stride_Emean_seqlen + pid_h * stride_Emean_head
    
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # hdim
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # dstate
    offs_k = tl.arange(0, BLOCK_SIZE_K)   # chunk_size
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_hdim + offs_k[None, :] * stride_dout_seqlen)
    # E_ptrs = E_ptr + (offs_m[:, None] * stride_E_hdim + offs_k[None, :] * stride_E_seqlen)
    # I_ptrs = I_ptr + (offs_m[:, None] * stride_I_hdim + offs_k[None, :] * stride_I_seqlen)
    # E_mean_ptrs = E_mean_ptr +  offs_k[None, :] * stride_Emean_seqlen
    c_ptrs = c_ptr + (offs_n[None, :] * stride_c_dstate + offs_k[:, None] * stride_c_seqlen)
    
    dA_cumsum_ptrs = dA_cumsum_ptr + (offs_k[None, :] * stride_dA_cs_csize + offs_m[:, None] * stride_dA_cs_hdim)
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0).to(tl.float32)
        # E = tl.load(E_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0).to(tl.float32)
        # I = tl.load(I_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0).to(tl.float32)
        # E_mean = tl.load(E_mean_ptrs, mask=(offs_k < chunk_size_limit - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale_k = tl.exp(dA_cs_k)
        else:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
            scale_k = tl.where(seq_idx_k == seq_idx_prev, tl.exp(dA_cs_k), 0.0)
        
        # we differentiate through the mask here
        # sigma = E - I - E_mean
        # sigma = tl.sigmoid(sigma)
        # dout *= sigma
        # we also calculate the loss here wrt E and I and E_mean
        
        
        
        dout = (dout * scale_k).to(dout_ptr.dtype.element_ty)
        c = tl.load(c_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0)
        acc += tl.dot(dout, c)
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        c_ptrs += BLOCK_SIZE_K * stride_c_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    out = acc.to(dprev_states_ptr.dtype.element_ty)

    dprev_states_ptr += pid_b * stride_dprev_states_batch + pid_c * stride_dprev_states_chunk + pid_h * stride_dprev_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dprev_states_ptrs = dprev_states_ptr + (offs_m[:, None] * stride_dprev_states_hdim + offs_n[None, :] * stride_dprev_states_dstate)
    tl.store(dprev_states_ptrs, out, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate))


#dr
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=2, num_warps=4),
    ],
    key=['chunk_size', 'dstate', 'IS_TRITON_22'],
)
@triton.jit
def _chunk_scan_bwd_dC_dr_kernel(
    # Inputs
    dout_ptr, M_ptr, states_ptr, C_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Outputs
    dC_ptr, ddA_prev_ptr,
    # Dimensions
    chunk_size, hdim, dstate, batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides for dout (b, l, h, p)
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    # Strides for M (b, l, h, p) - precomputed mask
    stride_M_batch, stride_M_seqlen, stride_M_head, stride_M_hdim,
    # Strides for states (b, c, h, p, n)
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    # Strides for C (b, l, g, n)
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    # Strides for dA_cumsum (b, h, c, l, p)
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
    # Strides for output dC (b, l, nsplits, g, n)
    stride_dC_batch, stride_dC_seqlen, stride_dC_split, stride_dC_group, stride_dC_dstate,
    # Strides for output ddA_prev (b, l, h, p)
    stride_ddA_batch, stride_ddA_seqlen, stride_ddA_head, stride_ddA_hdim,
    # Seq idx
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta
    HAS_DDA_CS: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n  # time block
    pid_n = tl.program_id(axis=0) % num_pid_n   # dstate block
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    mask_time = offs_m < chunk_size_limit
    
    # Output pointers
    dC_ptr += pid_b * stride_dC_batch + pid_c * chunk_size * stride_dC_seqlen + pid_g * stride_dC_group + pid_s * stride_dC_split
    dC_ptrs = dC_ptr + (offs_m[:, None] * stride_dC_seqlen + offs_n[None, :] * stride_dC_dstate)
    
    if HAS_DDA_CS:
        ddA_prev_ptr += pid_b * stride_ddA_batch + pid_c * chunk_size * stride_ddA_seqlen
    
    # Accumulators
    acc_dC = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        acc_ddA = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Base pointers for inputs (offset by batch and chunk)
    base_dout_ptr = dout_ptr + pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen
    base_M_ptr = M_ptr + pid_b * stride_M_batch + pid_c * chunk_size * stride_M_seqlen
    base_C_ptr = C_ptr + pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + pid_g * stride_C_head
    base_dA_cs_ptr = dA_cumsum_ptr + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk
    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk
    
    # Load states for this chunk (we need all headdims for reduction)
    # states shape: (h, p, n)
    
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    
    for h_iter in range(nheads_iter):
        h = pid_g * (nheads // ngroups) + pid_s * nheads_per_program + h_iter
        
        # Pointers for this head
        dout_ptrs = base_dout_ptr + h * stride_dout_head + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
        M_ptrs = base_M_ptr + h * stride_M_head + (offs_m[:, None] * stride_M_seqlen + offs_k[None, :] * stride_M_hdim)
        
        # states for this head: (p, n)
        states_ptrs = states_ptr + h * stride_states_head + (offs_k[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
        
        # ddA output pointer for this head
        if HAS_DDA_CS:
            ddA_ptrs_h = ddA_prev_ptr + h * stride_ddA_head + (offs_m[:, None] * stride_ddA_seqlen + offs_k[None, :] * stride_ddA_hdim)
        
        # Reduction loop over headdim (p dimension)
        for k_start in range(0, hdim, BLOCK_SIZE_K):
            k = k_start + offs_k
            mask_k = k < hdim
            
            # Load dout and precomputed M
            dout = tl.load(dout_ptrs, mask=mask_time[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            M = tl.load(M_ptrs, mask=mask_time[:, None] & mask_k[None, :], other=0.0)
            
            # Apply M to dout (because y_inter = M * ..., so backprop through M)
            dout_masked = dout * M
            
            # Load decay exp(S_t) for this headdim
            dA_cs = tl.load(base_dA_cs_ptr + h * stride_dA_cs_head + offs_m[:, None] * stride_dA_cs_csize + k[None, :] * stride_dA_cs_hdim,
                           mask=mask_time[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            decay = tl.exp(dA_cs)
            
            # Scale dout by decay
            dout_scaled = dout_masked * decay  # (time, headdim)
            
            # Load states: (headdim, dstate) - need transposed or careful indexing
            # states[k, n] where k is headdim block
            states = tl.load(states_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < dstate), other=0.0)
            states = states.to(tl.float32)  # Cast to float32 to match dout_scaled
            
            # dC contribution: dout_scaled^T @ states? No
            # dout_scaled: (time, headdim), states: (headdim, dstate)
            # We want: dC[time, dstate] += sum_headdim(dout_scaled[time, h] * states[h, dstate])
            # That's a direct matmul
            acc_dC += tl.dot(dout_scaled, states)
            
            # ddA_prev contribution (if needed)
            if HAS_DDA_CS:
                # C @ states^T gives us the y_inter_component before decay and M
                # Load C for this time and dstate: (time, dstate)
                C_load_ptrs = base_C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_n[None, :] * stride_C_dstate)
                C_val = tl.load(C_load_ptrs, mask=mask_time[:, None] & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
                
                # Compute C @ states^T -> (time, headdim)
                y_component = tl.dot(C_val, tl.trans(states))  # (t, n) @ (n, k) = (t, k)
                
                # ddA contribution: dout_masked * decay * y_component? 
                # Actually ddA = dout * M * exp(S) * (C @ h)
                # dout_masked is dout * M, so:
                contrib = dout_scaled * y_component  # (time, headdim) * (time, headdim)
                
                # Accumulate to ddA (atomic add because multiple blocks may write)
                tl.atomic_add(ddA_ptrs_h + k_start, contrib, mask=mask_time[:, None] & mask_k[None, :])
            
            # Advance pointers
            dout_ptrs += BLOCK_SIZE_K * stride_dout_hdim
            M_ptrs += BLOCK_SIZE_K * stride_M_hdim
            states_ptrs += BLOCK_SIZE_K * stride_states_hdim
    
    # Store dC
    tl.store(dC_ptrs, acc_dC.to(dC_ptr.dtype.element_ty), mask=mask_time[:, None] & (offs_n[None, :] < dstate))


#dr
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['chunk_size', 'IS_TRITON_22'],
)
@triton.jit
def _chunk_scan_bwd_dcb_dr_kernel(
    # Pointers to inputs - now uses x_masked and M instead of x, E, I, E_mean
    x_masked_ptr, M_ptr, dout_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Pointer to output
    dCB_ptr,
    # Dimensions
    chunk_size, hdim, batch, seqlen,
    # Strides for x_masked (b, l, h, p)
    stride_xm_batch, stride_xm_seqlen, stride_xm_head, stride_xm_hdim,
    # Strides for M (b, l, h, p)
    stride_M_batch, stride_M_seqlen, stride_M_head, stride_M_hdim,
    # Strides for dout (b, l, h, p)
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    # Strides for dA_cumsum (b, h, c, l, p)
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
    # Strides for output dCB (b, c, g, l, l)
    stride_dCB_batch, stride_dCB_chunk, stride_dCB_head, stride_dCB_csize_m, stride_dCB_csize_n,
    # Seq idx strides
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    """
    Compute dCB[i,j] = sum_p dout[i,p] * M[j,p] * x_masked[j,p] * exp(A[i,p] - A[j,p])
    for intra-chunk path. Uses precomputed x_masked and M.
    """
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_g = tl.program_id(axis=2)
    
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n  # row i (dest time)
    pid_n = tl.program_id(axis=0) % num_pid_n   # col j (source time)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # i (dest)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # j (source)
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # headdim p (reduction)
    
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    # Early exit for upper triangle (causal)
    if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
        dCB_ptr += pid_b * stride_dCB_batch + pid_c * stride_dCB_chunk + pid_g * stride_dCB_head
        dCB_ptrs = dCB_ptr + (offs_m[:, None] * stride_dCB_csize_m + offs_n[None, :] * stride_dCB_csize_n)
        tl.store(dCB_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dCB_ptr.dtype.element_ty),
                mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
        return
    
    # Offset pointers by batch, chunk
    base_xm_ptr = x_masked_ptr + pid_b * stride_xm_batch + pid_c * chunk_size * stride_xm_seqlen
    base_M_ptr = M_ptr + pid_b * stride_M_batch + pid_c * chunk_size * stride_M_seqlen
    base_dout_ptr = dout_ptr + pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen
    base_dA_cs_ptr = dA_cumsum_ptr + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_g * stride_dA_cs_head
    
    dCB_ptr += pid_b * stride_dCB_batch + pid_c * stride_dCB_chunk + pid_g * stride_dCB_head
    
    # Accumulator for dCB
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Reduction loop over headdim
    k_max = (hdim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    for k_block in range(k_max):
        k_start = k_block * BLOCK_SIZE_K
        k = k_start + offs_k
        mask_k = k < hdim
        
        # --- Load precomputed x_masked and M for source position j ---
        # x_masked already contains x * M, so we just need to load it
        xm_ptrs_j = base_xm_ptr + pid_g * stride_xm_head + (offs_n[:, None] * stride_xm_seqlen + k[None, :] * stride_xm_hdim)
        M_ptrs_j = base_M_ptr + pid_g * stride_M_head + (offs_n[:, None] * stride_M_seqlen + k[None, :] * stride_M_hdim)
        
        x_masked_j = tl.load(xm_ptrs_j, mask=(offs_n[:, None] < chunk_size_limit) & mask_k[None, :], other=0.0).to(tl.float32)
        # M_j not strictly needed since x_masked already incorporates the mask, but keep for consistency
        
        # --- Load dout for dest position i ---
        dout_ptrs_i = base_dout_ptr + pid_g * stride_dout_head + (offs_m[:, None] * stride_dout_seqlen + k[None, :] * stride_dout_hdim)
        dout_i = tl.load(dout_ptrs_i, mask=(offs_m[:, None] < chunk_size_limit) & mask_k[None, :], other=0.0).to(tl.float32)
        
        # --- Load and compute decay exp(min(A_i - A_j, 0)) ---
        dA_cs_i_ptrs = base_dA_cs_ptr + offs_m[:, None] * stride_dA_cs_csize + k[None, :] * stride_dA_cs_hdim
        dA_cs_j_ptrs = base_dA_cs_ptr + offs_n[:, None] * stride_dA_cs_csize + k[None, :] * stride_dA_cs_hdim
        
        dA_cs_i = tl.load(dA_cs_i_ptrs, mask=(offs_m[:, None] < chunk_size) & mask_k[None, :], other=0.0).to(tl.float32)
        dA_cs_j = tl.load(dA_cs_j_ptrs, mask=(offs_n[:, None] < chunk_size) & mask_k[None, :], other=0.0).to(tl.float32)
        
        # Simple vectorized approach: use average decay across K dimension
        dA_cs_i_mean = tl.sum(dA_cs_i, axis=1) / BLOCK_SIZE_K  # (M,)
        dA_cs_j_mean = tl.sum(dA_cs_j, axis=1) / BLOCK_SIZE_K  # (N,)
        decay_avg = tl.exp(tl.minimum(dA_cs_i_mean[:, None] - dA_cs_j_mean[None, :], 0.0))  # (M, N)
        
        # Compute weighted outer product: sum_k dout[m,k] * x_masked[n,k] with decay
        contrib = tl.dot(dout_i.to(x_masked_j.dtype), tl.trans(x_masked_j))  # (M, N)
        acc += contrib * decay_avg
    
    # Apply causal mask (lower triangular)
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    acc = tl.where(causal_mask, acc, 0.0)
    
    # Store dCB
    dCB_ptrs = dCB_ptr + (offs_m[:, None] * stride_dCB_csize_m + offs_n[None, :] * stride_dCB_csize_n)
    tl.store(dCB_ptrs, acc.to(dCB_ptr.dtype.element_ty), 
            mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


#dr
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_DSTATE': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_DSTATE': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_DSTATE': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_DSTATE': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_DSTATE': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_DSTATE': 64}, num_stages=3, num_warps=8),
    ],
    key=['chunk_size', 'hdim', 'dstate', 'IS_TRITON_22'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_dr_kernel(
    # Pointers to inputs - using precomputed x_masked and M
    x_masked_ptr, M_ptr, dout_ptr, dstates_ptr, B_ptr, CB_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Pointers to outputs
    dx_ptr, dMask_ptr,
    # Dimensions
    chunk_size, hdim, dstate, batch, seqlen, nheads_ngroups_ratio,
    # Strides for x_masked (b, l, h, p)
    stride_xm_batch, stride_xm_seqlen, stride_xm_head, stride_xm_hdim,
    # Strides for M (b, l, h, p)
    stride_M_batch, stride_M_seqlen, stride_M_head, stride_M_hdim,
    # Strides for dout (b, l, h, p)
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    # Strides for dstates (b, c, h, p, n)
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    # Strides for B (b, l, g, n)
    stride_B_batch, stride_B_seqlen, stride_B_head, stride_B_dstate,
    # Strides for CB (b, c, g, l, l)
    stride_CB_batch, stride_CB_chunk, stride_CB_head, stride_CB_csize_m, stride_CB_csize_k,
    # Strides for dA_cumsum (b, h, c, l, p)
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
    # Strides for output dx (b, c, l, h, p)
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    # Strides for output dMask (b, c, l, h, p)
    stride_dMask_batch, stride_dMask_seqlen, stride_dMask_head, stride_dMask_hdim,
    # Strides for seq_idx (b, l)
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    # Offset pointers by batch, chunk, and head - using precomputed x_masked and M
    x_masked_ptr += pid_b * stride_xm_batch + pid_c * chunk_size * stride_xm_seqlen + pid_h * stride_xm_head
    M_ptr += pid_b * stride_M_batch + pid_c * chunk_size * stride_M_seqlen + pid_h * stride_M_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    B_ptr += pid_b * stride_B_batch + pid_c * chunk_size * stride_B_seqlen + (pid_h // nheads_ngroups_ratio) * stride_B_head
    CB_ptr += pid_b * stride_CB_batch + pid_c * stride_CB_chunk + (pid_h // nheads_ngroups_ratio) * stride_CB_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dMask_ptr += pid_b * stride_dMask_batch + pid_c * chunk_size * stride_dMask_seqlen + pid_h * stride_dMask_head
    
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    # Tile offsets within chunk
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # time dimension
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # headdim dimension
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    # --- Load precomputed M ---
    M_ptrs = M_ptr + offs_m[:, None] * stride_M_seqlen + offs_n[None, :] * stride_M_hdim
    M = tl.load(M_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0)
    
    # Accumulator for gradient w.r.t. masked input
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # --- Source 1: Gradient from state path (via dstates) ---
    # Load dA_cumsum for decay calculation: shape (time, headdim)
    dA_cs_m_ptrs = dA_cumsum_ptr + offs_m[:, None] * stride_dA_cs_csize + offs_n[None, :] * stride_dA_cs_hdim
    dA_cs_m = tl.load(dA_cs_m_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    
    # Load dA_cumsum at end of chunk for the decay factor exp(S_end - S_t)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize + offs_n[None, :] * stride_dA_cs_hdim, 
                         mask=offs_n[None, :] < hdim, other=0.0).to(tl.float32)
    
    # Compute decay: exp(S_last - S_t) with numerical stability
    decay = tl.exp(tl.minimum(dA_cs_last - dA_cs_m, 0.0))
    
    # Matrix multiply: B * dstates^T -> (time, dstate) @ (dstate, headdim) = (time, headdim)
    offs_k = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    B_ptrs = B_ptr + (offs_m[:, None] * stride_B_seqlen + offs_k[None, :] * stride_B_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_k[:, None] * stride_dstates_dstate)
    
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        B = tl.load(B_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(B_ptr.dtype.element_ty)
        acc = tl.dot(B, dstates) * decay
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            B = tl.load(B_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(B_ptr.dtype.element_ty)
            acc += tl.dot(B, dstates)
            B_ptrs += BLOCK_SIZE_K * stride_B_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= decay

    # --- Source 2: Gradient from local CB path (intra-chunk) ---
    # For per-channel A decay: dx[m, n] = Σ_k CB[m, k] * exp(S_m[n] - S_k[n]) * dout[k, n]
    # The decay depends on (m, k, n) - a 3D tensor. 
    #
    # Hybrid approach: Load data in blocks for coalescing, unroll inner k-loop for exact decay.
    # This gives O(K_MAX / BLOCK_SIZE_K * BLOCK_SIZE_K) = O(K_MAX) iterations, but with
    # coalesced loads every BLOCK_SIZE_K iterations.
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Load dA_cumsum for rows m: shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m[:, None] * stride_dA_cs_csize + offs_n[None, :] * stride_dA_cs_hdim,
                      mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    
    # Causal upper bound: only positions k <= m contribute
    K_MAX = min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    
    for k_base in range(0, K_MAX, BLOCK_SIZE_K):
        # Load CB block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        CB_block = tl.load(CB_ptr + offs_m[:, None] * stride_CB_csize_m + (k_base + offs_k[None, :]) * stride_CB_csize_k,
                           mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k_base), other=0.0).to(tl.float32)
        
        # Load dout block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        dout_block = tl.load(dout_ptr + (k_base + offs_k[:, None]) * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim,
                             mask=(offs_k[:, None] < K_MAX - k_base) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        
        # Load dA_cumsum for rows k: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        dA_cs_k_block = tl.load(dA_cumsum_ptr + (k_base + offs_k[:, None]) * stride_dA_cs_csize + offs_n[None, :] * stride_dA_cs_hdim,
                                mask=(offs_k[:, None] < K_MAX - k_base) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        
        # Unroll over k positions within the block for exact per-channel decay
        # Process each k in the block: contrib[m, n] = CB[m, k] * exp(S_m[n] - S_k[n]) * dout[k, n]
        for k_idx in range(BLOCK_SIZE_K):
            k_pos = k_base + k_idx
            
            # Skip if beyond K_MAX (will be masked anyway, but saves compute)
            # Note: can't early break in Triton, so we rely on masking
            
            # Causal mask: only k <= m contributes
            causal_mask = k_pos <= offs_m  # (BLOCK_SIZE_M,)
            valid_k = k_idx < (K_MAX - k_base)
            
            # Extract k-th column/row from loaded blocks
            # CB_block[:, k_idx] -> (BLOCK_SIZE_M,)
            cb_col = tl.sum(tl.where(offs_k[None, :] == k_idx, CB_block, 0.0), axis=1)
            
            # dout_block[k_idx, :] -> (BLOCK_SIZE_N,)
            dout_row = tl.sum(tl.where(offs_k[:, None] == k_idx, dout_block, 0.0), axis=0)
            
            # dA_cs_k_block[k_idx, :] -> (BLOCK_SIZE_N,)
            dA_cs_k = tl.sum(tl.where(offs_k[:, None] == k_idx, dA_cs_k_block, 0.0), axis=0)
            
            # Compute per-channel decay: exp(S_m[n] - S_k[n])
            # dA_cs_m: (BLOCK_SIZE_M, BLOCK_SIZE_N), dA_cs_k: (BLOCK_SIZE_N,)
            decay = tl.exp(tl.minimum(dA_cs_m - dA_cs_k[None, :], 0.0))  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
            
            # Contribution: cb_col[m] * decay[m, n] * dout_row[n]
            contrib = cb_col[:, None] * decay * dout_row[None, :]  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
            
            # Apply causal mask and validity mask
            acc += tl.where(causal_mask[:, None] & valid_k, contrib, 0.0)

    # --- Gradient Split: dX and dM ---
    # Load precomputed x_masked
    x_masked_ptrs = x_masked_ptr + offs_m[:, None] * stride_xm_seqlen + offs_n[None, :] * stride_xm_hdim
    x_masked = tl.load(x_masked_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    
    # Get x from x_masked / M (avoiding division by zero where M=0)
    # Where M=0, x_masked=0 anyway, so we can safely compute x = x_masked / (M + eps) 
    # or just use x_masked directly in dM calculation since dM_add should be zero where M=0
    # Actually for dM_add = dX_tilde * x, where M=0, x_masked=0, so we need original x.
    # We can recover: x = x_masked where M=1, else irrelevant since gradient doesn't flow there
    # Use safe division: x_reconstructed = x_masked / (M + 1e-6)
    M_float = M.to(tl.float32)
    x = tl.where(M_float > 0.5, x_masked / M_float, 0.0)
    
    # dX_tilde is gradient w.r.t. masked input X_tilde = M * X
    dX_tilde = acc
    
    # Gradient w.r.t. original X: only flows where M=1
    dX = dX_tilde * M_float
    
    # Gradient w.r.t. mask M from input side: dM_input = dX_tilde * X
    dM_add = dX_tilde * x
    
    # Store dX
    dx_ptrs = dx_ptr + offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim
    tl.store(dx_ptrs, dX.to(dx_ptr.dtype.element_ty), mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    
    # Accumulate to dMask (atomic add - dMask already contains dM_output from top of backward)
    dMask_ptrs = dMask_ptr + offs_m[:, None] * stride_dMask_seqlen + offs_n[None, :] * stride_dMask_hdim
    tl.atomic_add(dMask_ptrs, dM_add, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

#dr - optimized version using precomputed x_masked and M
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
)
@triton.jit
def _chunk_scan_fwd_dr_kernel(
    # Pointers to matrices - now uses x_masked and M instead of E, I, E_mean
    cb_ptr, x_masked_ptr, z_ptr, out_ptr, out_x_ptr, M_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, prev_states_ptr, D_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_M_batch, stride_M_seqlen, stride_M_head, stride_M_hdim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_hdim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head,
    THRESHOLD: tl.constexpr,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    OUTPUT_ACTIVATION: tl.constexpr,
):
    """
    Optimized DR_SSM chunk scan kernel using precomputed x_masked and M.
    Eliminates redundant E, I, E_mean loads and sigmoid computations.
    """
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    
    # Setup pointers - now M instead of E, I, E_mean
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_masked_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    
    # Save base pointer for M (needed for output mask)
    M_base_ptr = M_ptr + pid_b * stride_M_batch + pid_c * chunk_size * stride_M_seqlen + pid_h * stride_M_head
    
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load dA_cumsum at output positions: (BLOCK_M, BLOCK_N)
    dA_cs_m_ptrs = dA_cumsum_ptr + offs_m[:, None] * stride_dA_cs_csize + offs_n[None, :] * stride_dA_cs_hdim
    dA_cs_m = tl.load(dA_cs_m_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # y_inter-chunk contribution
    if IS_TRITON_22 or pid_c > -1:
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
        prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
        
        if not HAS_SEQ_IDX:
            scale_m_inner = tl.exp(dA_cs_m)
        else:
            scale_m_inner = tl.where(seq_idx_m[:, None] == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        
        # Load precomputed mask M instead of computing from E, I, E_mean
        M_inner = tl.load(M_base_ptr + offs_m[:, None] * stride_M_seqlen + offs_n[None, :] * stride_M_hdim, 
                          mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc = tl.dot(C, prev_states) * scale_m_inner * M_inner
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m_inner * M_inner

    # Within-chunk contribution loop - use precomputed x_masked directly
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_masked_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k[None, :] * stride_dA_cs_csize + offs_n[:, None] * stride_dA_cs_hdim
    
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        # Load CB: (BLOCK_M, BLOCK_K)
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
        
        # Load precomputed x_masked: (BLOCK_K, BLOCK_N) - already has mask applied
        x_masked = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim), other=0.0)
        
        # Load dA_cs_k: (BLOCK_N, BLOCK_K) - per headdim
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=(offs_n[:, None] < hdim) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
        
        # Compute decay: (BLOCK_M, BLOCK_N, BLOCK_K)
        decay = tl.exp(tl.minimum(dA_cs_m[:, :, None] - dA_cs_k[None, :, :], 0.0))
        
        # Apply decay to x_masked
        x_transposed = tl.trans(x_masked, 1, 0)  # (BLOCK_N, BLOCK_K)
        x_broadcast = x_transposed[None, :, :]  # (1, BLOCK_N, BLOCK_K)
        x_decayed = decay * x_broadcast
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = offs_m[:, None, None] >= (k + offs_k[None, None, :])
            x_decayed = tl.where(causal_mask, x_decayed, 0.0)
        
        # Expand cb and compute contribution
        cb = cb.to(x_masked_ptr.dtype.element_ty)
        cb_expanded = cb[:, None, :]
        temp = cb_expanded * x_decayed
        acc += tl.sum(temp, axis=2)
        
        # Increment pointers
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # Apply output mask using precomputed M
    M_out_ptrs = M_base_ptr + (offs_m[:, None] * stride_M_seqlen + offs_n[None, :] * stride_M_hdim)
    output_mask = tl.load(M_out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    acc = acc * output_mask
    
    # ========================================= finished main computation =========================================
    
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Skip connection D - note: using x_masked for residual
    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(x_masked_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
                             mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc += x_residual * D

    # Optional ReLU
    if OUTPUT_ACTIVATION == "relu":
        acc = tl.where(acc > THRESHOLD, acc, 0.0)

    # Z gating (SiLU)
    if HAS_Z:
        out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    # Store output
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))


#dr
def _chunk_scan_bwd_dr_dz(x, z, out, dout, chunk_size, has_ddAcs=True, D=None, dz=None, recompute_output=False):
    batch, seqlen, nheads, headdim = x.shape
    assert z.shape == x.shape
    assert out.shape == x.shape
    assert dout.shape == out.shape
    nchunks = math.ceil(seqlen / chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
    if has_ddAcs:
        ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=x.device, dtype=torch.float32)
    if D is not None:
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    if dz is not None:
        assert dz.shape == z.shape
    else:
        dz = torch.empty_like(z)
    if recompute_output:
        outz = torch.empty_like(x)
    dout_x = torch.empty_like(dout)
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    grid_dz = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dz_kernel[grid_dz](
            dout, out, z, x, D, outz if recompute_output else None,
            dz, dout_x, dD, ddA_cumsum if has_ddAcs else None,
            chunk_size, headdim,
            batch, seqlen,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            D.stride(0) if D is not None else 0,
            *((outz.stride(0), outz.stride(1), outz.stride(2), outz.stride(3)) if recompute_output else (0, 0, 0, 0)),
            dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3),
            dout_x.stride(0), dout_x.stride(1), dout_x.stride(2), dout_x.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            *((ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3))
              if has_ddAcs else (0, 0, 0, 0)),
            D is not None,
            D.dim() == 2 if D is not None else True,
            has_ddAcs,
            BLOCK_SIZE_N=max(triton.next_power_of_2(headdim), 16),
            RECOMPUTE_OUTPUT=recompute_output,
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_bwd_dz_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return_vals = (dz, dout_x, dD, ddA_cumsum) if has_ddAcs else (dz, dout_x, dD)
    return return_vals if not recompute_output else (*return_vals, outz)




#dr
#dr
def _chunk_scan_bwd_dC_dr(
    dout, M, states, C, dA_cumsum, seq_idx=None,
    dC=None, ddA_prev=None, has_ddA=False
):
    """
    Compute gradients w.r.t. C and ddA_prev from inter-chunk path.
    
    Args:
        dout: (batch, seqlen, nheads, headdim) - gradient from upstream (dout_ssm)
        M: (batch, seqlen, nheads, headdim) - precomputed binary mask
        states: (batch, nchunks, nheads, headdim, dstate) - forward states (h_prev)
        C: (batch, seqlen, ngroups, dstate)
        dA_cumsum: (batch, nheads, nchunks, chunk_size, headdim)
        
    Returns:
        dC: (batch, seqlen, ngroups, dstate)
        ddA_prev: (batch, seqlen, nheads, headdim) if has_ddA else None
    """
    batch, seqlen, nheads, headdim = dout.shape
    _, nchunks, _, _, dstate = states.shape
    _, _, ngroups, _ = C.shape
    
    assert nheads % ngroups == 0
    nheads_ngroups_ratio = nheads // ngroups
    
    # Setup splits for parallelization
    sm_count = torch.cuda.get_device_properties(dout.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    
    if dC is None:
        dC = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=dout.device, dtype=torch.float32)
    
    if has_ddA:
        if ddA_prev is None:
            ddA_prev = torch.empty(batch, seqlen, nheads, headdim, device=dout.device, dtype=torch.float32)
        else:
            assert ddA_prev.shape == (batch, seqlen, nheads, headdim)
    else:
        ddA_prev = None
    
    chunk_size = dA_cumsum.shape[3]
    
    grid = lambda META: (
        triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
        batch * nchunks,
        nsplits * ngroups
    )
    
    IS_TRITON_22 = int(triton.__version__.split('.')[1]) >= 2 if hasattr(triton, '__version__') else 0
    
    with torch.cuda.device(dout.device.index):
        _chunk_scan_bwd_dC_dr_kernel[grid](
            dout, M, states, C, dA_cumsum, seq_idx,
            dC, ddA_prev,
            chunk_size, headdim, dstate, batch, seqlen, nheads, nheads_per_program, ngroups,
            # dout strides
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            # M strides
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            # states strides (b, c, h, p, n)
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            # C strides (b, l, g, n)
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            # dA_cumsum strides (b, h, c, l, p)
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
            # dC strides (b, l, s, g, n)
            dC.stride(0), dC.stride(1), dC.stride(2), dC.stride(3), dC.stride(4),
            # ddA strides
            *((ddA_prev.stride(0), ddA_prev.stride(1), ddA_prev.stride(2), ddA_prev.stride(3)) if has_ddA else (0,0,0,0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_DDA_CS=has_ddA,
            HAS_SEQ_IDX=seq_idx is not None,
            IS_TRITON_22=IS_TRITON_22,
        )
    
    # Sum over splits
    dC = dC.sum(dim=2)
    
    return dC, ddA_prev

#dr

#dr

def _chunk_scan_bwd_dcb_dr(
    x_masked, M, dout, dA_cumsum, seq_idx=None, ngroups=1
):
    """
    Compute dCB gradient using precomputed x_masked and M.
    """
    batch, seqlen, nheads, headdim = x_masked.shape
    _, nchunks, _, chunk_size, _ = dA_cumsum.shape
    
    assert nheads % ngroups == 0
    
    # Output tensor
    dCB = torch.empty(batch, nchunks, ngroups, chunk_size, chunk_size, 
                     device=x_masked.device, dtype=torch.float32)
    
    grid = lambda META: (
        triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']),
        batch * nchunks,
        ngroups
    )
    
    IS_TRITON_22 = int(triton.__version__.split('.')[1]) >= 2 if hasattr(triton, '__version__') else 0
    
    with torch.cuda.device(x_masked.device.index):
        _chunk_scan_bwd_dcb_dr_kernel[grid](
            x_masked, M, dout, dA_cumsum, seq_idx,
            dCB,
            chunk_size, headdim, batch, seqlen,
            # x_masked strides (b, l, h, p)
            x_masked.stride(0), x_masked.stride(1), x_masked.stride(2), x_masked.stride(3),
            # M strides (b, l, h, p)
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            # dout strides (b, l, h, p)
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            # dA_cumsum strides (b, h, c, l, p)
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
            # dCB strides (b, c, g, l, l)
            dCB.stride(0), dCB.stride(1), dCB.stride(2), dCB.stride(3), dCB.stride(4),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
            IS_TRITON_22=IS_TRITON_22,
        )
    
    return dCB




#dr
def _chunk_scan_chunk_state_bwd_dx_dr(
    x_masked, M, dout, dstates, B, CB, dA_cumsum, seq_idx=None,
    dx=None, dMask=None
):
    """
    Compute dx using precomputed x_masked and M.
    """
    batch, seqlen, nheads, headdim = x_masked.shape
    _, nchunks, _, _, dstate = dstates.shape
    _, _, ngroups, chunk_size, _ = CB.shape
    
    assert M.shape == x_masked.shape
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size, headdim)
    assert dout.shape == x_masked.shape
    assert nheads % ngroups == 0, f"nheads {nheads} must be divisible by ngroups {ngroups}"
    
    # Allocate outputs if not provided
    if dx is None:
        dx = torch.empty_like(x_masked)
    else:
        assert dx.shape == x_masked.shape
        
    # dMask must be provided (contains dM_output from _chunk_scan_through_mask_bwd)
    assert dMask is not None
    assert dMask.shape == x_masked.shape
    
    # Rearrange tensors to chunked view: (b, l, h, p) -> (b, c, chunk_size, h, p)
    x_masked_chunk = rearrange(x_masked, "b (c l) h p -> b c l h p", c=nchunks)
    M_chunk = rearrange(M, "b (c l) h p -> b c l h p", c=nchunks)
    dout_chunk = rearrange(dout, "b (c l) h p -> b c l h p", c=nchunks)
    dx_chunk = rearrange(dx, "b (c l) h p -> b c l h p", c=nchunks)
    dMask_chunk = rearrange(dMask, "b (c l) h p -> b c l h p", c=nchunks)

    # Grid: each block handles a tile of (chunk_size, headdim)
    def grid(META):
        return (
            triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
            batch * nchunks,
            nheads
        )
    
    # Triton version detection
    IS_TRITON_22 = int(triton.__version__.split('.')[1]) >= 2 if hasattr(triton, '__version__') else 0
    
    # Launch kernel
    with torch.cuda.device(x_masked.device.index):
        _chunk_scan_chunk_state_bwd_dx_dr_kernel[grid](
            x_masked_chunk, M_chunk, dout_chunk, dstates, B, CB, dA_cumsum,
            seq_idx,
            dx_chunk, dMask_chunk,
            chunk_size, headdim, dstate, batch, seqlen, nheads // ngroups,
            # x_masked strides: (b, c, l, h, p)
            x_masked_chunk.stride(0), x_masked_chunk.stride(2), x_masked_chunk.stride(3), x_masked_chunk.stride(4),
            # M strides: same as x_masked
            M_chunk.stride(0), M_chunk.stride(2), M_chunk.stride(3), M_chunk.stride(4),
            # dout strides: same as x_masked
            dout_chunk.stride(0), dout_chunk.stride(2), dout_chunk.stride(3), dout_chunk.stride(4),
            # dstates strides: (b, c, h, p, n)
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            # B strides: (b, l, g, n) - l is global seqlen, need chunk view
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            # CB strides: (b, c, g, l, l) - last two are chunk_size dims
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4),
            # dA_cumsum strides: (b, h, c, l, p)
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
            # dx strides: same as x_masked_chunk
            dx_chunk.stride(0), dx_chunk.stride(2), dx_chunk.stride(3), dx_chunk.stride(4),
            # dMask strides: same as dMask_chunk
            dMask_chunk.stride(0), dMask_chunk.stride(2), dMask_chunk.stride(3), dMask_chunk.stride(4),
            # seq_idx strides: (b, l) or None
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
            IS_TRITON_22=IS_TRITON_22,
        )
    
    
    return dx  # dMask is modified in-place

#dr
def _chunk_scan_bwd_ddAcs_stable_dr(
    x_masked, M, dout, CB, dA_cumsum, seq_idx=None, ngroups=None
):
    """
    Compute ddA contribution from CB matrix path using stable algorithm.
    Uses precomputed x_masked and M.
    
    Returns ddA with shape (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = dout.shape
    _, nchunks, ngroups_actual, chunk_size, _ = CB.shape
    
    if ngroups is None:
        ngroups = ngroups_actual
        
    assert nheads % ngroups == 0
    
    # Output buffer
    ddA = torch.zeros(batch, seqlen, nheads, headdim, 
                     device=dout.device, dtype=torch.float32)
    
    # Grid: parallel over (batch*chunk, nheads, chunk_size)
    grid = (chunk_size, batch * nchunks, nheads)
    
    IS_TRITON_22 = int(triton.__version__.split('.')[1]) >= 2 if hasattr(triton, '__version__') else 0
    
    with torch.cuda.device(dout.device.index):
        _chunk_scan_bwd_ddAcs_stable_dr_kernel[grid](
            x_masked, M, dout, CB, dA_cumsum, seq_idx,
            ddA,
            chunk_size, headdim, batch, seqlen, ngroups,
            # x_masked strides (b, l, h, p)
            x_masked.stride(0), x_masked.stride(1), x_masked.stride(2), x_masked.stride(3),
            # M strides (b, l, h, p)
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            # dout (b, l, h, p)
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            # CB (b, c, g, l, l)
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4),
            # dA_cumsum (b, h, c, l, p)
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), 
            dA_cumsum.stride(3), dA_cumsum.stride(4),
            # ddA (b, l, h, p)
            ddA.stride(0), ddA.stride(1), ddA.stride(2), ddA.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    
    return ddA



#dr
def _chunk_scan_bwd_dr_dstates(C, dA_cumsum, dout, seq_idx=None, dtype=None):
    batch, seqlen, nheads, headdim = dout.shape
    # dA_cumsum has shape (batch, nheads, nchunks, chunk_size, headdim) for DR_SSM
    _, _, nchunks, chunk_size, _ = dA_cumsum.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size, headdim)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    dtype = C.dtype if dtype is None else dtype
    dprev_states = torch.empty(batch, nchunks, nheads, headdim, dstate, device=C.device, dtype=dtype)
    grid_dstates = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                            batch * nchunks, nheads)
    with torch.cuda.device(C.device.index):
        _chunk_scan_bwd_dr_dstates_kernel[grid_dstates](
            dout, C, dprev_states, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nchunks, nheads // ngroups,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            dprev_states.stride(0), dprev_states.stride(1), dprev_states.stride(2), dprev_states.stride(3), dprev_states.stride(4),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),dA_cumsum.stride(4),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return dprev_states



#dr
def _chunk_scan_dr_fwd(cb, x_masked, dA_cumsum, C, M, states, D=None, z=None, seq_idx=None, output_activation=None, threshold=0.0):
    """
    Forward pass for DR_SSM chunk scan using precomputed masked input and mask.
    
    Args:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size) - C @ B product
        x_masked: (batch, seqlen, nheads, headdim) - precomputed x * M  
        dA_cumsum: (batch, nheads, nchunks, chunk_size, headdim) - cumulative decay
        C: (batch, seqlen, ngroups, dstate) - C matrix
        M: (batch, seqlen, nheads, headdim) - precomputed binary mask
        states: (batch, nchunks, nheads, headdim, dstate) - chunk states
        D: optional skip connection
        z: optional gating
        seq_idx: optional sequence indices
        output_activation: optional activation
        threshold: activation threshold
        
    Returns:
        out: (batch, seqlen, nheads, headdim) - output
        out_x: optional pre-gating output
    """
    batch, seqlen, nheads, headdim = x_masked.shape
    _, _, ngroups, dstate = C.shape
    _, _, nchunks, chunk_size, _ = dA_cumsum.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert M.shape == (batch, seqlen, nheads, headdim)
    if z is not None:
        assert z.shape == x_masked.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size, headdim)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    
    # Allocate output
    out = torch.empty(batch, seqlen, nheads, headdim, device=x_masked.device, dtype=x_masked.dtype)
    if z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=x_masked.device, dtype=x_masked.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                         batch * nchunks, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3))
                 if z is not None else (0, 0, 0, 0))
    
    _chunk_scan_fwd_dr_kernel[grid](
        cb, x_masked, z, out, out_x, M, dA_cumsum, seq_idx, C, states, D,
        chunk_size, headdim, dstate,
        batch, seqlen, nheads // ngroups,
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
        x_masked.stride(0), x_masked.stride(1), x_masked.stride(2), x_masked.stride(3),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        M.stride(0), M.stride(1), M.stride(2), M.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
        D.stride(0) if D is not None else 0,
        threshold,
        True,  # IS_CAUSAL
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
        OUTPUT_ACTIVATION=output_activation,
    )
    return out, out_x



# =============================================================================================================================

class ChunkScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, C, x, dt, dA_cumsum, prev_states, D=None, z=None, output_activation=None, threshold = 0.0):
        # Check constraints.
        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        assert B.shape == (batch, seqlen, ngroups, dstate)
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen == nchunks * chunk_size
        assert C.shape == B.shape
        if z is not None:
            assert z.shape == x.shape
        if D is not None:
            assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
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
        CB = _bmm_chunk_fwd(C, B, chunk_size)
        out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, prev_states, D=D, z=z, output_activation=output_activation, threshold=threshold)
        ctx.save_for_backward(out if z is None else out_x, B, C, CB, x, dt, dA_cumsum, prev_states, D, z)
        ctx.output_activation = output_activation
        ctx.threshold = threshold
        return out

    @staticmethod
    def backward(ctx, dout):
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        out, B, C, CB, x, dt, dA_cumsum, prev_states, D, z = ctx.saved_tensors
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        _, _, ngroups, dstate = B.shape
        assert dout.shape == (batch, seqlen, nheads, headdim)
        if z is not None:
            dz, dout, dD, ddA_cumsum = _chunk_scan_bwd_dz(x, z, out, dout, chunk_size=chunk_size, D=D)
        else:
            dz = None
            if ctx.output_activation == 'relu':
                relu_mask = out > ctx.threshold     
                dout = dout * relu_mask
        dprev_states = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, dtype=prev_states.dtype)
        dC = _chunk_scan_bwd_dC(prev_states, dA_cumsum, dout, ngroups=ngroups)
        dC = dC.to(C.dtype)
        dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, ngroups=ngroups)
        dCB = dCB.to(CB.dtype)
        dB = _bmm_chunk_bwd(C, dCB)
        dC = _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC)
        dx, ddt = _chunk_scan_bwd_dx(CB, x, dt, dA_cumsum, dout, D=D)
        # Formula for ddA_cumsum, assuming out is the output of the forward pass before adding x * D.
        # ddA_cumsum = torch.einsum("bclhp,bclhp->bhcl", out.float(), dout.float()) - ddt * dt
        if z is not None:
            ddA_cumsum -= ddt * dt
        else: # If z is not None, we already calculated ddA_cumsum and dD when computing dz
            ddA_cumsum, dD = _chunk_scan_bwd_ddAcs_unstable(x, dt, out, dout, ddt, D=D)
        ddA_cumsum = ddA_cumsum.to(dA_cumsum.dtype)
        return dB, dC, dx, ddt, ddA_cumsum, dprev_states, dD, dz


def chunk_scan(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None, output_activation=None):
    """
    prev_states contains the initial_states at index 0, and the state for the next-to-last chunk at index -1.
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    return ChunkScanFn.apply(B, C, x, dt, dA_cumsum, prev_states, D, z, output_activation=output_activation)


def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)
