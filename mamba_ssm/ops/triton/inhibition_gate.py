import torch
from torch import Tensor
from packaging import version

import triton
import triton.language as tl

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

# ---------------------------------------------------------------------------
# Triton helpers
# ---------------------------------------------------------------------------

if TRITON3:
    @triton.jit
    def _softplus(x):
        return tl.where(x > 20.0, x, tl.math.log(tl.math.exp(x) + 1))
else:
    @triton.jit
    def _softplus(x):
        return tl.where(x > 20.0, x, tl.math.log1p(tl.exp(x)))


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _inhibition_mask_fwd_kernel(
    # Pointers
    X_ptr,      # Used to compute mask
    I_ptr,      # Inhibition
    INPUT_ptr,  # The actual input to apply mask to (can be X or Y)
    OUT_ptr,    # Output
    # Dimensions
    batch, seqlen, nheads, headdim,
    # Strides for X
    stride_xb, stride_xl, stride_xh, stride_xd,
    # Strides for I
    stride_ib, stride_il, stride_ih, stride_id,
    # Strides for INPUT
    stride_inb, stride_inl, stride_inh, stride_ind,
    # Strides for OUT
    stride_ob, stride_ol, stride_oh, stride_od,
    # Meta
    BLOCK_HD: tl.constexpr,
):
    """
    Computes: mask = (X - softplus(I) - softplus(mean(X)) > 0)
              output = INPUT * mask
    """
    pid = tl.program_id(0)
    pid_b = pid // (seqlen * nheads)
    rem = pid % (seqlen * nheads)
    pid_l = rem // nheads
    pid_h = rem % nheads

    offs_d = tl.arange(0, BLOCK_HD)
    mask_valid = offs_d < headdim

    # Base pointers
    x_base = X_ptr + pid_b * stride_xb + pid_l * stride_xl + pid_h * stride_xh
    i_base = I_ptr + pid_b * stride_ib + pid_l * stride_il + pid_h * stride_ih
    in_base = INPUT_ptr + pid_b * stride_inb + pid_l * stride_inl + pid_h * stride_inh
    o_base = OUT_ptr + pid_b * stride_ob + pid_l * stride_ol + pid_h * stride_oh

    # Load X and I (for mask computation)
    x = tl.load(x_base + offs_d * stride_xd, mask=mask_valid, other=0.0).to(tl.float32)
    inh = tl.load(i_base + offs_d * stride_id, mask=mask_valid, other=0.0).to(tl.float32)
    
    # Load the actual input to apply mask to
    input_val = tl.load(in_base + offs_d * stride_ind, mask=mask_valid, other=0.0).to(tl.float32)

    # Compute mask from X
    x_mean = tl.sum(x, axis=0) / headdim
    sp_i = _softplus(inh)
    sp_m = _softplus(x_mean)
    
    # Binary mask: (X - softplus(I) - softplus(mean(X)) > 0)
    pre = x - sp_i - sp_m
    binary_mask = (pre > 0.0).to(tl.float32)
    
    # Apply mask to input
    out = input_val * binary_mask
    
    tl.store(o_base + offs_d * stride_od, out, mask=mask_valid)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _inhibition_mask_bwd_kernel(
    # Pointers (inputs)
    X_ptr, I_ptr, INPUT_ptr, DOUT_ptr,
    # Pointers (outputs)
    DX_ptr, DI_ptr, DINPUT_ptr,
    # Dimensions
    batch, seqlen, nheads, headdim,
    # Strides for X
    stride_xb, stride_xl, stride_xh, stride_xd,
    # Strides for I
    stride_ib, stride_il, stride_ih, stride_id,
    # Strides for INPUT
    stride_inb, stride_inl, stride_inh, stride_ind,
    # Strides for DOUT
    stride_dob, stride_dol, stride_doh, stride_dod,
    # Strides for DX
    stride_dxb, stride_dxl, stride_dxh, stride_dxd,
    # Strides for DI
    stride_dib, stride_dil, stride_dih, stride_did,
    # Strides for DINPUT
    stride_dinb, stride_dinl, stride_dinh, stride_dind,
    # Meta
    BLOCK_HD: tl.constexpr,
):
    """
    Backward for inhibition gate with Gumbel-Softmax straight-through.
    
    Forward uses BINARY mask (sparse activations).
    Backward uses:
    - BINARY mask for d_INPUT (matches forward, prevents spikes)
    - SIGMOID for d_X and d_I (smooth gradients, enables learning)
    """
    pid = tl.program_id(0)
    pid_b = pid // (seqlen * nheads)
    rem = pid % (seqlen * nheads)
    pid_l = rem // nheads
    pid_h = rem % nheads

    offs_d = tl.arange(0, BLOCK_HD)
    mask_valid = offs_d < headdim

    # Load X, I, INPUT, dout
    x_base = X_ptr + pid_b * stride_xb + pid_l * stride_xl + pid_h * stride_xh
    i_base = I_ptr + pid_b * stride_ib + pid_l * stride_il + pid_h * stride_ih
    in_base = INPUT_ptr + pid_b * stride_inb + pid_l * stride_inl + pid_h * stride_inh
    do_base = DOUT_ptr + pid_b * stride_dob + pid_l * stride_dol + pid_h * stride_doh

    x = tl.load(x_base + offs_d * stride_xd, mask=mask_valid, other=0.0).to(tl.float32)
    inh = tl.load(i_base + offs_d * stride_id, mask=mask_valid, other=0.0).to(tl.float32)
    input_val = tl.load(in_base + offs_d * stride_ind, mask=mask_valid, other=0.0).to(tl.float32)
    dout = tl.load(do_base + offs_d * stride_dod, mask=mask_valid, other=0.0).to(tl.float32)

    # Recompute forward intermediates
    x_mean = tl.sum(x, axis=0) / headdim
    sp_i = _softplus(inh)
    sp_m = _softplus(x_mean)
    pre = x - sp_i - sp_m
    
    # Temperature for sigmoid
    temperature = 5.0
    
    # Compute BOTH masks
    binary_mask = (pre > 0.0).to(tl.float32)  # For d_input
    soft_mask = _sigmoid(pre / temperature)    # For d_pre
    
    # Gradient w.r.t INPUT: use BINARY mask (matches forward)
    d_input = dout * binary_mask
    
    # Gradient w.r.t mask parameters: use SIGMOID derivative
    d_soft_mask = dout * input_val
    sigmoid_deriv = soft_mask * (1.0 - soft_mask) / temperature
    d_pre = d_soft_mask * sigmoid_deriv
    
    # Gradient w.r.t I (through softplus)
    sig_i = _sigmoid(inh)
    d_i = -d_pre * sig_i
    
    # Gradient w.r.t X (through mean and softplus)
    sig_m = _sigmoid(x_mean)
    sum_dpre = tl.sum(d_pre, axis=0)
    d_x = d_pre - (sig_m * sum_dpre) / headdim
    
    # Store gradients
    dx_base = DX_ptr + pid_b * stride_dxb + pid_l * stride_dxl + pid_h * stride_dxh
    di_base = DI_ptr + pid_b * stride_dib + pid_l * stride_dil + pid_h * stride_dih
    din_base = DINPUT_ptr + pid_b * stride_dinb + pid_l * stride_dinl + pid_h * stride_dinh

    tl.store(dx_base + offs_d * stride_dxd, d_x, mask=mask_valid)
    tl.store(di_base + offs_d * stride_did, d_i, mask=mask_valid)
    tl.store(din_base + offs_d * stride_dind, d_input, mask=mask_valid)
# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _inhibition_gate_fwd(x: Tensor, inh: Tensor, input_tensor: Tensor, out: Tensor = None) -> Tensor:
    """
    Forward: Compute binary mask from X, apply to input_tensor
    
    mask = (X - softplus(I) - softplus(mean(X, dim=-1)) > 0)
    output = input_tensor * mask
    
    Args:
        x:            (batch, seqlen, nheads, headdim) - used to compute mask
        inh:          (batch, seqlen, nheads, headdim) - inhibition vector I
        input_tensor: (batch, seqlen, nheads, headdim) - the tensor to apply mask to (can be X or Y)
        out:          optional pre-allocated output
    
    Returns:
        out: (batch, seqlen, nheads, headdim)
    """
    assert x.shape == inh.shape == input_tensor.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    if inh.stride(-1) != 1:
        inh = inh.contiguous()
    if input_tensor.stride(-1) != 1:
        input_tensor = input_tensor.contiguous()

    batch, seqlen, nheads, headdim = x.shape
    if out is None:
        out = torch.empty_like(input_tensor)
    assert out.stride(-1) == 1

    BLOCK_HD = triton.next_power_of_2(headdim)
    num_warps = 4 if BLOCK_HD >= 128 else (2 if BLOCK_HD >= 64 else 1)
    grid = (batch * seqlen * nheads,)
    
    with torch.cuda.device(x.device.index):
        _inhibition_mask_fwd_kernel[grid](
            x, inh, input_tensor, out,
            batch, seqlen, nheads, headdim,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            inh.stride(0), inh.stride(1), inh.stride(2), inh.stride(3),
            input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_HD=BLOCK_HD,
            num_warps=num_warps,
        )
    return out


def _inhibition_gate_bwd(
    x: Tensor, inh: Tensor, input_tensor: Tensor, dout: Tensor,
    dx: Tensor = None, di: Tensor = None, dinput: Tensor = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Backward for inhibition mask gate.
    
    Args:
        x:            (batch, seqlen, nheads, headdim) - used to compute mask
        inh:          (batch, seqlen, nheads, headdim) - inhibition vector
        input_tensor: (batch, seqlen, nheads, headdim) - the tensor mask was applied to
        dout:         (batch, seqlen, nheads, headdim) - gradient w.r.t. output
    
    Returns:
        dx:     (batch, seqlen, nheads, headdim) - gradient w.r.t. X
        di:     (batch, seqlen, nheads, headdim) - gradient w.r.t. I
        dinput: (batch, seqlen, nheads, headdim) - gradient w.r.t. input_tensor
    """
    assert x.shape == inh.shape == input_tensor.shape == dout.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    if inh.stride(-1) != 1:
        inh = inh.contiguous()
    if input_tensor.stride(-1) != 1:
        input_tensor = input_tensor.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()

    batch, seqlen, nheads, headdim = x.shape
    if dx is None:
        dx = torch.empty_like(x)
    if di is None:
        di = torch.empty_like(inh)
    if dinput is None:
        dinput = torch.empty_like(input_tensor)

    BLOCK_HD = triton.next_power_of_2(headdim)
    num_warps = 4 if BLOCK_HD >= 128 else (2 if BLOCK_HD >= 64 else 1)
    grid = (batch * seqlen * nheads,)
    
    with torch.cuda.device(x.device.index):
        _inhibition_mask_bwd_kernel[grid](
            x, inh, input_tensor, dout,
            dx, di, dinput,
            batch, seqlen, nheads, headdim,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            inh.stride(0), inh.stride(1), inh.stride(2), inh.stride(3),
            input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            di.stride(0), di.stride(1), di.stride(2), di.stride(3),
            dinput.stride(0), dinput.stride(1), dinput.stride(2), dinput.stride(3),
            BLOCK_HD=BLOCK_HD,
            num_warps=num_warps,
        )
    return dx, di, dinput


# ---------------------------------------------------------------------------
# PyTorch autograd wrapper
# ---------------------------------------------------------------------------

class InhibitionMaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inh, input_tensor):
        """
        Args:
            x: tensor used to compute mask
            inh: inhibition tensor
            input_tensor: tensor to apply mask to (can be same as x, or different like y)
        """
        out = _inhibition_gate_fwd(x, inh, input_tensor)
        ctx.save_for_backward(x, inh, input_tensor)
        return out
    
    @staticmethod
    def backward(ctx, dout):
        x, inh, input_tensor = ctx.saved_tensors
        dx, di, dinput = _inhibition_gate_bwd(x, inh, input_tensor, dout)
        return dx, di, dinput


def inhibition_mask(x: Tensor, inh: Tensor, input_tensor: Tensor) -> Tensor:
    """
    Compute binary mask from X and apply to input_tensor.
    
    mask = (X - softplus(I) - softplus(mean(X, dim=-1)) > 0)
    output = input_tensor * mask
    
    Args:
        x:            (batch, seqlen, nheads, headdim) - used to compute mask
        inh:          (batch, seqlen, nheads, headdim) - inhibition vector
        input_tensor: (batch, seqlen, nheads, headdim) - tensor to apply mask to
    
    Returns:
        output: (batch, seqlen, nheads, headdim)
    
    Examples:
        # Apply mask computed from X to X itself
        out = inhibition_mask(x, inh, x)
        
        # Apply mask computed from X to a different tensor Y
        out = inhibition_mask(x, inh, y)
    """
    return InhibitionMaskFunction.apply(x, inh, input_tensor)