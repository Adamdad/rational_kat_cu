import torch
import triton
import triton.language as tl

# --------------------
# Forward kernel
# --------------------
# The forward kernel computes for each element:
#   P = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5   (computed by Horner’s method)
#   Q = 1 + |b0|*|x| + |b1|*|x|^2 + |b2|*|x|^3 + |b3|*|x|^4
#   result = P / Q
#
# Each “group” uses 6 coefficients from a and 4 coefficients from b.
#
# We assume the following inputs:
#   x_ptr: pointer to input tensor (flattened, size = B*L*D)
#   a_ptr: pointer to numerator coefficients (per–group, groups = group count)
#   b_ptr: pointer to denominator coefficients (per–group)
#   result_ptr: pointer to output tensor (flattened)
#   x_size: total number of elements
#   D: size of the last dimension
#   D_per_group: D divided by the number of groups
#
# The grid is 1D.
BLOCK_SIZE = 256

@triton.jit
def rational_fwd_kernel(x_ptr, a_ptr, b_ptr, result_ptr,
                        x_size: tl.constexpr, D: tl.constexpr, D_per_group: tl.constexpr):
    # Compute the global index for each program instance
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < x_size

    # Load the input
    x_val = tl.load(x_ptr + offs, mask=mask)

    # Compute the “channel” index (d_index) and group index
    d_index = offs % D
    # integer division (since D_per_group divides D)
    g_index = d_index // D_per_group

    # Each group uses 6 numerator and 4 denominator coefficients.
    a_offset = g_index * 6
    b_offset = g_index * 4

    # Load coefficients for a.
    # (Each thread loads its group’s coefficients; this is not optimal if many threads share the same group,
    #  but it mirrors the original kernel logic.)
    s_a0 = tl.load(a_ptr + a_offset + 0)
    s_a1 = tl.load(a_ptr + a_offset + 1)
    s_a2 = tl.load(a_ptr + a_offset + 2)
    s_a3 = tl.load(a_ptr + a_offset + 3)
    s_a4 = tl.load(a_ptr + a_offset + 4)
    s_a5 = tl.load(a_ptr + a_offset + 5)

    # Load and take the absolute value for coefficients for b.
    s_b0 = tl.abs(tl.load(b_ptr + b_offset + 0))
    s_b1 = tl.abs(tl.load(b_ptr + b_offset + 1))
    s_b2 = tl.abs(tl.load(b_ptr + b_offset + 2))
    s_b3 = tl.abs(tl.load(b_ptr + b_offset + 3))

    # Compute absolute value of x for Q.
    abs_x = tl.abs(x_val)

    # Compute polynomial P using Horner’s method:
    # P = s_a0 + s_a1*x + s_a2*x^2 + ... + s_a5*x^5.
    # We do the computation in reverse order.
    P = s_a5
    P = tl.fma(P, x_val, s_a4)
    P = tl.fma(P, x_val, s_a3)
    P = tl.fma(P, x_val, s_a2)
    P = tl.fma(P, x_val, s_a1)
    P = tl.fma(P, x_val, s_a0)

    # Compute polynomial Q using Horner’s method:
    # Q = 1 + s_b0*|x| + s_b1*|x|^2 + s_b2*|x|^3 + s_b3*|x|^4.
    Q = s_b3
    Q = tl.fma(Q, abs_x, s_b2)
    Q = tl.fma(Q, abs_x, s_b1)
    Q = tl.fma(Q, abs_x, s_b0)
    Q = tl.fma(Q, abs_x, 1.0)

    # Write the result.
    tl.store(result_ptr + offs, P / Q, mask=mask)


def rational_fwd(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, group: int) -> torch.Tensor:
    """
    Forward computation: applies the rational function element‐wise.
      x: input tensor of shape [B, L, D]
      a: tensor with numerator coefficients (shape: [group, 6])
      b: tensor with denominator coefficients (shape: [group, 4])
    """
    assert x.is_contiguous()  # expects flattened memory view
    result = torch.empty_like(x)
    x_size = x.numel()
    D = x.shape[-1]
    D_per_group = D // group

    grid = lambda meta: (triton.cdiv(x_size, BLOCK_SIZE),)

    rational_fwd_kernel[grid](
        x,
        a,
        b,
        result,
        x_size,
        D,
        D_per_group,
    )
    return result


# --------------------
# Backward kernel
# --------------------
# The backward kernel computes gradients with respect to the input x and the coefficients.
# For each element it computes:
#
#   xp = x
#   axp = |x|
#   P = a0 + a1*x + a2*x^2 + ... + a5*x^5
#   Q = 1 + |b0|*axp + |b1|*axp^2 + |b2|*axp^3 + |b3|*axp^4
#   R = a1 + 2*a2*x + 3*a3*x^2 + 4*a4*x^3 + 5*a5*x^4
#   S = sign(x) * (|b0| + 2*|b1|*axp + 3*|b2|*axp^2 + 4*|b3|*axp^3)
#
# and then:
#   d_x = (R/Q + S * (-P/(Q^2))) * grad_o
#
# It also computes per–coefficient gradients:
#
#   d_a[0] = grad_o/Q,  d_a[i] = (x^i * grad_o)/Q, for i = 1,...,5
#   d_b[i] = (-P/(Q^2)) * sign(b[i]) * (axp^(i+1)) * grad_o, for i = 0,...,3
#
# The results for d_a and d_b are accumulated via atomic adds.
@triton.jit
def rational_bwd_kernel(grad_output_ptr, x_ptr, a_ptr, b_ptr,
                        d_x_ptr, d_a_ptr, d_b_ptr,
                        x_size: tl.constexpr, D: tl.constexpr, D_per_group: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < x_size

    # Load grad_output and x.
    grad_o = tl.load(grad_output_ptr + offs, mask=mask)
    x_val = tl.load(x_ptr + offs, mask=mask)

    # Determine group index
    d_index = offs % D
    g_index = d_index // D_per_group
    a_offset = g_index * 6
    b_offset = g_index * 4

    # Load coefficients for a.
    a0 = tl.load(a_ptr + a_offset + 0)
    a1 = tl.load(a_ptr + a_offset + 1)
    a2 = tl.load(a_ptr + a_offset + 2)
    a3 = tl.load(a_ptr + a_offset + 3)
    a4 = tl.load(a_ptr + a_offset + 4)
    a5 = tl.load(a_ptr + a_offset + 5)

    # Load coefficients for b (and compute their absolute values).
    b0 = tl.load(b_ptr + b_offset + 0)
    b1 = tl.load(b_ptr + b_offset + 1)
    b2 = tl.load(b_ptr + b_offset + 2)
    b3 = tl.load(b_ptr + b_offset + 3)
    b0_abs = tl.abs(b0)
    b1_abs = tl.abs(b1)
    b2_abs = tl.abs(b2)
    b3_abs = tl.abs(b3)

    # Compute powers of x.
    xp = x_val
    xp2 = xp * xp
    xp3 = xp2 * xp
    xp4 = xp3 * xp
    xp5 = xp4 * xp

    # Compute absolute value of x and its powers.
    axp = tl.abs(x_val)
    axp2 = axp * axp
    axp3 = axp2 * axp
    axp4 = axp3 * axp

    # Compute P, Q, R, S.
    P = a0 + a1 * xp + a2 * xp2 + a3 * xp3 + a4 * xp4 + a5 * xp5
    Q = 1.0 + b0_abs * axp + b1_abs * axp2 + b2_abs * axp3 + b3_abs * axp4
    R = a1 + 2.0*a2 * xp + 3.0*a3 * xp2 + 4.0*a4 * xp3 + 5.0*a5 * xp4
    # Compute sign(x): if x<0 then -1, else 1.
    sign_x = tl.where(x_val < 0, -1.0, 1.0)
    S = sign_x * (b0_abs + 2.0*b1_abs * axp + 3.0*b2_abs * axp2 + 4.0*b3_abs * axp3)

    mpq2 = -P / (Q * Q)

    # Compute gradient for x.
    dx = (R / Q + S * mpq2) * grad_o
    tl.store(d_x_ptr + offs, dx, mask=mask)

    # Compute gradients for a.
    da0 = grad_o / Q
    da1 = xp * grad_o / Q
    da2 = xp2 * grad_o / Q
    da3 = xp3 * grad_o / Q
    da4 = xp4 * grad_o / Q
    da5 = xp5 * grad_o / Q

    # Compute gradients for b.
    # Note: for each coefficient b_i, we use the original sign.
    sign_b0 = tl.where(b0 < 0, -1.0, 1.0)
    sign_b1 = tl.where(b1 < 0, -1.0, 1.0)
    sign_b2 = tl.where(b2 < 0, -1.0, 1.0)
    sign_b3 = tl.where(b3 < 0, -1.0, 1.0)
    db0 = mpq2 * sign_b0 * axp * grad_o
    db1 = mpq2 * sign_b1 * axp2 * grad_o
    db2 = mpq2 * sign_b2 * axp3 * grad_o
    db3 = mpq2 * sign_b3 * axp4 * grad_o

    # Accumulate contributions for coefficients.
    # (Each thread’s computed gradients are atomically added to global memory.)
    # For each lane in the block, we perform an atomic add.
    for i, da in enumerate([da0, da1, da2, da3, da4, da5]):
        tl.atomic_add(d_a_ptr + (a_offset + i), da, mask=mask)
    for i, db in enumerate([db0, db1, db2, db3]):
        tl.atomic_add(d_b_ptr + (b_offset + i), db, mask=mask)


def rational_bwd(grad_output: torch.Tensor,
                 x: torch.Tensor,
                 a: torch.Tensor,
                 b: torch.Tensor,
                 group: int):
    """
    Backward computation.
      grad_output: gradient of the loss with respect to the output, shape [B, L, D]
      x: input tensor of shape [B, L, D]
      a: tensor with numerator coefficients (shape: [group, 6])
      b: tensor with denominator coefficients (shape: [group, 4])
    Returns a tuple: (d_x, d_a, d_b)
      d_x: gradient with respect to x, same shape as x.
      d_a: gradient with respect to a (same shape as a, float type).
      d_b: gradient with respect to b (same shape as b, float type).
    """
    d_x = torch.empty_like(x)
    # Initialize gradients for coefficients with zeros.
    d_a = torch.zeros_like(a, dtype=torch.float32)
    d_b = torch.zeros_like(b, dtype=torch.float32)

    x_size = x.numel()
    D = x.shape[-1]
    D_per_group = D // group

    grid = lambda meta: (triton.cdiv(x_size, BLOCK_SIZE),)

    rational_bwd_kernel[grid](
        grad_output,
        x,
        a,
        b,
        d_x,
        d_a,
        d_b,
        x_size,
        D,
        D_per_group,
    )
    return d_x, d_a, d_b

