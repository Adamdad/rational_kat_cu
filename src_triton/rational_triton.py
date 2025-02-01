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
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < x_size

    x = tl.load(x_ptr + idx, mask=mask)
    d_index = idx % D
    g_index = d_index // D_per_group

    a_idx = g_index * 6
    b_idx = g_index * 4

    s_a = tl.load(a_ptr + a_idx + tl.arange(0, 6), mask=tl.arange(0, 6) < 6)
    s_b = tl.load(b_ptr + b_idx + tl.arange(0, 4), mask=tl.arange(0, 4) < 4)
    s_b = tl.abs(s_b)

    xp1 = x
    abs_xp1 = tl.abs(xp1)

    # Compute P using Horner's method
    P = s_a[5]
    for i in range(4, -1, -1):
        P = P * xp1 + s_a[i]

    # Compute Q using Horner's method
    Q = s_b[3]
    for i in range(2, -1, -1):
        Q = Q * abs_xp1 + s_b[i]
    Q = Q * abs_xp1 + 1.0

    result = P / Q
    tl.store(result_ptr + idx, result, mask=mask)

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

    s_a = tl.load(a_ptr + a_idx + tl.arange(0, 6), mask=tl.arange(0, 6) < 6)
    s_b = tl.load(b_ptr + b_idx + tl.arange(0, 4), mask=tl.arange(0, 4) < 4)
    s_b_abs = tl.abs(s_b)

    xp = x
    axp = tl.abs(xp)

    xp_powers = tl.zeros((5,), dtype=tl.float32)
    xp_powers = tl.set(xp_powers, 0, xp)
    for i in range(1, 5):
        xp_powers = tl.set(xp_powers, i, xp_powers[i-1] * xp)

    axp_powers = tl.zeros((4,), dtype=tl.float32)
    axp_powers = tl.set(axp_powers, 0, axp)
    for i in range(1, 4):
        axp_powers = tl.set(axp_powers, i, axp_powers[i-1] * axp)

    P = s_a[0] + s_a[1] * xp_powers[0] + s_a[2] * xp_powers[1] + s_a[3] * xp_powers[2] + s_a[4] * xp_powers[3] + s_a[5] * xp_powers[4]
    Q = 1.0 + s_b_abs[0] * axp_powers[0] + s_b_abs[1] * axp_powers[1] + s_b_abs[2] * axp_powers[2] + s_b_abs[3] * axp_powers[3]
    R = s_a[1] + 2.0 * s_a[2] * xp_powers[0] + 3.0 * s_a[3] * xp_powers[1] + 4.0 * s_a[4] * xp_powers[2] + 5.0 * s_a[5] * xp_powers[3]
    S = tl.sign(xp) * (s_b_abs[0] + 2.0 * s_b_abs[1] * axp_powers[0] + 3.0 * s_b_abs[2] * axp_powers[1] + 4.0 * s_b_abs[3] * axp_powers[2])

    grad_o = tl.load(grad_output_ptr + idx, mask=mask)
    mpq2 = -P / (Q * Q)
    d_i_x = (R / Q + S * mpq2) * grad_o
    tl.store(d_x_ptr + idx, d_i_x, mask=mask)

    local_da = tl.zeros((6,), dtype=tl.float32)
    local_db = tl.zeros((4,), dtype=tl.float32)

    local_da = tl.set(local_da, 0, 1.0 / Q * grad_o)
    for i in range(1, 6):
        local_da = tl.set(local_da, i, xp_powers[i-1] / Q * grad_o)

    for i in range(0, 4):
        local_db = tl.set(local_db, i, mpq2 * tl.sign(s_b[i]) * axp_powers[i] * grad_o)

    for i in range(0, 6):
        tl.atomic_add(d_a_ptr + a_idx + i, local_da[i])
    for i in range(0, 4):
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