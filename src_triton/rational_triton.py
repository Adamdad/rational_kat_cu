import torch
import triton
import triton.language as tl

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

@triton.jit
def rational_bwd_kernel(
    grad_output_ptr, x_ptr, a_ptr, b_ptr,
    d_x_ptr, d_a_ptr, d_b_ptr,
    B, L, D, group, x_size, n_size, d_size, D_per_group,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < x_size

    x = tl.load(x_ptr + idx, mask=mask)
    d_index = idx % D
    g_index = d_index // D_per_group

    a_idx = g_index * 6
    b_idx = g_index * 4

    # Load coefficients for a (6 elements)
    s_a = tl.zeros((6,), dtype=tl.float32)
    for i in tl.static_range(6):  # Use static_range for compile-time loop
        s_a[i] = tl.load(a_ptr + a_idx + i)

    # Load coefficients for b (4 elements)
    s_b = tl.zeros((4,), dtype=tl.float32)
    for i in tl.static_range(4):  # Use static_range for compile-time loop
        s_b[i] = tl.load(b_ptr + b_idx + i)
        
    s_b_abs = tl.abs(s_b)

    xp = x
    axp = tl.abs(xp)

    # Compute powers of xp
    xp_powers = tl.zeros((5,), dtype=tl.float32)
    xp_powers[0] = xp
    for i in tl.static_range(1, 5):  # Use static_range for compile-time loop
        xp_powers[i] = xp_powers[i-1] * xp

    # Compute powers of axp
    axp_powers = tl.zeros((4,), dtype=tl.float32)
    axp_powers[0] = axp
    for i in tl.static_range(1, 4):  # Use static_range for compile-time loop
        axp_powers[i] = axp_powers[i-1] * axp

    # Compute P, Q, R, S
    P = s_a[0] + s_a[1] * xp_powers[0] + s_a[2] * xp_powers[1] + s_a[3] * xp_powers[2] + s_a[4] * xp_powers[3] + s_a[5] * xp_powers[4]
    Q = 1.0 + s_b_abs[0] * axp_powers[0] + s_b_abs[1] * axp_powers[1] + s_b_abs[2] * axp_powers[2] + s_b_abs[3] * axp_powers[3]
    R = s_a[1] + 2.0 * s_a[2] * xp_powers[0] + 3.0 * s_a[3] * xp_powers[1] + 4.0 * s_a[4] * xp_powers[2] + 5.0 * s_a[5] * xp_powers[3]
    S = tl.sign(xp) * (s_b_abs[0] + 2.0 * s_b_abs[1] * axp_powers[0] + 3.0 * s_b_abs[2] * axp_powers[1] + 4.0 * s_b_abs[3] * axp_powers[2])

    grad_o = tl.load(grad_output_ptr + idx, mask=mask)
    mpq2 = -P / (Q * Q)
    d_i_x = (R / Q + S * mpq2) * grad_o
    tl.store(d_x_ptr + idx, d_i_x, mask=mask)

    # Compute gradients for a and b
    local_da = tl.zeros((6,), dtype=tl.float32)
    local_db = tl.zeros((4,), dtype=tl.float32)

    local_da[0] = 1.0 / Q * grad_o
    for i in tl.static_range(1, 6):  # Use static_range for compile-time loop
        local_da[i] = xp_powers[i-1] / Q * grad_o

    for i in tl.static_range(0, 4):  # Use static_range for compile-time loop
        local_db[i] = mpq2 * tl.sign(s_b[i]) * axp_powers[i] * grad_o

    # Accumulate gradients for a and b
    for i in tl.static_range(0, 6):  # Use static_range for compile-time loop
        tl.atomic_add(d_a_ptr + a_idx + i, local_da[i])
    for i in tl.static_range(0, 4):  # Use static_range for compile-time loop
        tl.atomic_add(d_b_ptr + b_idx + i, local_db[i])
        
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