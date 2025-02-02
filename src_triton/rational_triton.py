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
@triton.jit
def rational_fwd_kernel(
    x_ptr, a_ptr, b_ptr, result_ptr,
    B, L, D, group, x_size, D_per_group,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < x_size

    # Load input elements.
    x_val = tl.load(x_ptr + offs, mask=mask)

    # Determine d index and group index.
    d_index = offs % D
    g_index = d_index // D_per_group

    # Compute coefficient offsets.
    a_offset = g_index * 6
    b_offset = g_index * 4

    # Load numerator coefficients.
    s_a0 = tl.load(a_ptr + a_offset + 0)
    s_a1 = tl.load(a_ptr + a_offset + 1)
    s_a2 = tl.load(a_ptr + a_offset + 2)
    s_a3 = tl.load(a_ptr + a_offset + 3)
    s_a4 = tl.load(a_ptr + a_offset + 4)
    s_a5 = tl.load(a_ptr + a_offset + 5)

    # Load denominator coefficients (using absolute value).
    s_b0 = tl.abs(tl.load(b_ptr + b_offset + 0))
    s_b1 = tl.abs(tl.load(b_ptr + b_offset + 1))
    s_b2 = tl.abs(tl.load(b_ptr + b_offset + 2))
    s_b3 = tl.abs(tl.load(b_ptr + b_offset + 3))

    abs_x = tl.abs(x_val)

    # Compute numerator polynomial P(x) via Horner's method.
    P = s_a5
    P = tl.fma(P, x_val, s_a4)
    P = tl.fma(P, x_val, s_a3)
    P = tl.fma(P, x_val, s_a2)
    P = tl.fma(P, x_val, s_a1)
    P = tl.fma(P, x_val, s_a0)

    # Compute denominator polynomial Q(x).
    Q = s_b3
    Q = tl.fma(Q, abs_x, s_b2)
    Q = tl.fma(Q, abs_x, s_b1)
    Q = tl.fma(Q, abs_x, s_b0)
    Q = tl.fma(Q, abs_x, 1.0)

    tl.store(result_ptr + offs, P / Q, mask=mask)

def rational_fwd_triton(x, n, d, group):
    B, L, D = x.shape
    x_size = x.numel()
    D_per_group = D // group

    result = torch.empty_like(x)
    BLOCK_SIZE = 256
    num_blocks = (x_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    rational_fwd_kernel[(num_blocks,)](
        x, n, d, result,
        B, L, D, group, x_size, D_per_group,
        BLOCK_SIZE=BLOCK_SIZE
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
def rational_bwd_kernel(
    grad_output_ptr, x_ptr, a_ptr, b_ptr,
    d_x_ptr, d_a_ptr, d_b_ptr,
    B, L, D, group, x_size, n_size, d_size, D_per_group,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread's global index.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < x_size

    # Allocate shared memory for accumulating coefficient gradients.
    # For numerator: group * 6 coefficients.
    # For denominator: group * 4 coefficients.
    # (Assuming group is small; otherwise, adjust the allocation.)
    shared_da = tl.zeros((32 * 6,), dtype=tl.float32)  # 32 is a safe upper bound for group.
    shared_db = tl.zeros((32 * 4,), dtype=tl.float32)

    # Compute each thread's local gradients.
    grad_o = tl.load(grad_output_ptr + offs, mask=mask)
    x_val = tl.load(x_ptr + offs, mask=mask)

    # Determine group index: each element belongs to group = (d_index // D_per_group)
    d_index = offs % D
    g_index = d_index // D_per_group
    a_offset = g_index * 6  # Offset into the coefficients arrays.
    b_offset = g_index * 4

    # Load coefficients for a.
    a0 = tl.load(a_ptr + a_offset + 0)
    a1 = tl.load(a_ptr + a_offset + 1)
    a2 = tl.load(a_ptr + a_offset + 2)
    a3 = tl.load(a_ptr + a_offset + 3)
    a4 = tl.load(a_ptr + a_offset + 4)
    a5 = tl.load(a_ptr + a_offset + 5)

    # Load coefficients for b and compute their absolute values.
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
    R = a1 + 2.0 * a2 * xp + 3.0 * a3 * xp2 + 4.0 * a4 * xp3 + 5.0 * a5 * xp4

    sign_x = tl.where(x_val < 0, -1.0, 1.0)
    S = sign_x * (b0_abs + 2.0 * b1_abs * axp + 3.0 * b2_abs * axp2 + 4.0 * b3_abs * axp3)
    mpq2 = -P / (Q * Q)

    # Compute gradient for x.
    dx = (R / Q + S * mpq2) * grad_o
    tl.store(d_x_ptr + offs, dx, mask=mask)

    # Compute gradients for coefficients.
    # For numerator:
    da0 = grad_o / Q
    da1 = xp * grad_o / Q
    da2 = xp2 * grad_o / Q
    da3 = xp3 * grad_o / Q
    da4 = xp4 * grad_o / Q
    da5 = xp5 * grad_o / Q
    # For denominator (using original sign for b).
    sign_b0 = tl.where(b0 < 0, -1.0, 1.0)
    sign_b1 = tl.where(b1 < 0, -1.0, 1.0)
    sign_b2 = tl.where(b2 < 0, -1.0, 1.0)
    sign_b3 = tl.where(b3 < 0, -1.0, 1.0)
    db0 = mpq2 * sign_b0 * axp * grad_o
    db1 = mpq2 * sign_b1 * axp2 * grad_o
    db2 = mpq2 * sign_b2 * axp3 * grad_o
    db3 = mpq2 * sign_b3 * axp4 * grad_o

    # --- Accumulate into shared memory ---
    # Each thread atomically adds its local gradients into the per-block shared memory.
    # Note: We use an offset into the shared arrays based on the group index.
    sm_offset_a = g_index * 6
    tl.atomic_add(shared_da + (sm_offset_a + 0), da0, mask=mask)
    tl.atomic_add(shared_da + (sm_offset_a + 1), da1, mask=mask)
    tl.atomic_add(shared_da + (sm_offset_a + 2), da2, mask=mask)
    tl.atomic_add(shared_da + (sm_offset_a + 3), da3, mask=mask)
    tl.atomic_add(shared_da + (sm_offset_a + 4), da4, mask=mask)
    tl.atomic_add(shared_da + (sm_offset_a + 5), da5, mask=mask)

    sm_offset_b = g_index * 4
    tl.atomic_add(shared_db + (sm_offset_b + 0), db0, mask=mask)
    tl.atomic_add(shared_db + (sm_offset_b + 1), db1, mask=mask)
    tl.atomic_add(shared_db + (sm_offset_b + 2), db2, mask=mask)
    tl.atomic_add(shared_db + (sm_offset_b + 3), db3, mask=mask)

    # Ensure all threads in the block have finished updating shared memory.
    tl.barrier()

    # Have only one thread per block write the accumulated shared memory to global memory.
    if tl.program_id(axis=0) % BLOCK_SIZE == 0:
        # For each possible group in this block, write the shared accumulations.
        # (Here we assume that the number of groups is small, so we iterate over all groups.)
        for g in range(group):
            for i in range(6):
                val = tl.load(shared_da + (g * 6 + i))
                # Use atomic add to accumulate block results into global memory.
                tl.atomic_add(d_a_ptr + (g * 6 + i), val)
            for i in range(4):
                val = tl.load(shared_db + (g * 4 + i))
                tl.atomic_add(d_b_ptr + (g * 4 + i), val)
        
def rational_bwd_triton(grad_output, x, n, d, group):
    B, L, D = x.shape
    x_size = x.numel()
    n_size = n.numel()
    d_size = d.numel()
    D_per_group = D // group

    d_x = torch.empty_like(x)
    d_n = torch.zeros_like(n, dtype=torch.float32)
    d_d = torch.zeros_like(d, dtype=torch.float32)

    BLOCK_SIZE = 256
    num_blocks = (x_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    rational_bwd_kernel[(num_blocks,)](
        grad_output, x, n, d,
        d_x, d_n, d_d,
        B, L, D, group, x_size, n_size, d_size, D_per_group,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return d_x, d_n, d_d


class rational_triton_1dgroup(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight_numerator, weight_denominator, group):
        """
        Forward pass of the custom autograd function.
        
        Args:
            ctx: Context object used to stash information for backward computation.
            input (Tensor): Input tensor.
            weight_numerator (Tensor): Weights of the numerator polynomial.
            weight_denominator (Tensor): Weights of the denominator polynomial.
            group (int): The group number.

        Returns:
            Tensor: The result of the rational function applied to the input tensor.
        """
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        ctx.group = group
        x = rational_fwd_triton(input, weight_numerator, weight_denominator, group)
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        """
        Backward pass of the custom autograd function.
        
        Args:
            ctx: Context object from the forward pass.
            grad_output (Tensor): Gradient of the output tensor.

        Returns:
            tuple: Gradients of the input, weight_numerator, weight_denominator.
        """
        input, weight_numerator, weight_denominator = ctx.saved_tensors
        group = ctx.group
        d_input, d_weight_numerator, d_weight_denominator = rational_bwd_triton(grad_output, input, weight_numerator, weight_denominator, group)
        return d_input, d_weight_numerator, d_weight_denominator, None
