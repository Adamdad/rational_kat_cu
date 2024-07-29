#include <torch/extension.h>

template <typename scalar_t>
__global__ void rational_fwd_cuda_kernel_1dgroup(
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, 
    scalar_t* __restrict__ result, 
    int B, int L, int D, int group, int total_elements, int D_per_group) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;  // Prevent out-of-bounds memory access

    // Calculate the index within the dimension D
    int d_index = idx % D;
    // Calculate the group index based on the position within dimension D
    int g_index = d_index / D_per_group;

    scalar_t s_a[6];
    scalar_t s_b[4];
    for (int i = 0; i < 6; ++i) {
        s_a[i] = a[g_index * 6 + i];
    }

    for (int i = 0; i < 4; ++i) {
        s_b[i] = b[g_index * 4 + i];
    }

    // Obtain the input value from the tensor
    scalar_t xp1 = x[idx];
    scalar_t abs_xp1 = abs(xp1);

    // Compute the polynomial for P using Horner's method
    scalar_t P = s_a[5];
    for (int i = 4; i >= 0; --i) {
        P = fmaf(P, xp1, s_a[i]);
    }
    
    // Compute the polynomial for Q using Horner's method
    scalar_t Q = s_b[3];
    for (int i = 2; i >= 0; --i) {
        Q = fmaf(Q, abs_xp1, s_b[i]);
    }
    Q = fmaf(Q, abs_xp1, 1.0);

    result[idx] = P / Q;
}

torch::Tensor rational_fwd_cuda_1dgroup(
    torch::Tensor x, 
    torch::Tensor n, 
    torch::Tensor d,
    int group
    ){
    auto result = at::empty_like(x);
    const int total_elements = x.numel();
    int threads_per_block = 256;  // Adjust as needed based on device capabilities
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rational_fwd_cuda_1dgroup", ([&] {
    rational_fwd_cuda_kernel_1dgroup<scalar_t>
        <<<num_blocks, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x.size(0), x.size(1), x.size(2), group, total_elements, x.size(2) / group);
        }));

    return result;
}

//P(X) = a_0 + a_1*X + a_2*X^2 ...
//Q(X) = 1 + |b_0||X| + |b_1||X|^2 + |b_2||X|^3
//R(X) = a_1 + 2*a_2*X + 3*a_3*X ...
//S(X) = sign(X) * ( |b_0| + 2|b_1||X| + 3|b_2||X|^2 ...)
//dF/dx = (-P(X)/Q(X)^2)*S(X) + R(X)/Q(X)
//dF/da_i = x^i/Q(X), i \in {0,5}
//dF/db_i = (-P(X)/Q(X)^2) * sign(b_i) * |X^{i+1}| , i \in {0,4}


template <typename scalar_t>
__global__ void rational_bwd_cuda_kernel_1dgroup(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {
    
    __shared__ double sda[6];
    __shared__ double sdb[4];

    double local_da[6] = {0}; // Local accumulation arrays
    double local_db[4] = {0};

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < x_size) {
        scalar_t xp = x[index];
        scalar_t axp = abs(xp);
        // Compute powers of xp
        scalar_t xp_powers[5];
        xp_powers[0] = xp;
        xp_powers[1] = xp * xp_powers[0]; // xp^2
        xp_powers[2] = xp * xp_powers[1]; // xp^3
        xp_powers[3] = xp * xp_powers[2]; // xp^4
        xp_powers[4] = xp * xp_powers[3]; // xp^5

        // Compute powers of axp
        scalar_t axp_powers[4];
        axp_powers[0] = axp;
        axp_powers[1] = axp * axp_powers[0]; // axp^2
        axp_powers[2] = axp * axp_powers[1]; // axp^3
        axp_powers[3] = axp * axp_powers[2]; // axp^4

        // Compute absolute values once
        scalar_t b0_abs = abs(b[0]);
        scalar_t b1_abs = abs(b[1]);
        scalar_t b2_abs = abs(b[2]);
        scalar_t b3_abs = abs(b[3]);

        scalar_t P = a[0] + a[1] * xp_powers[0] + a[2] * xp_powers[1] + a[3] * xp_powers[2] + a[4] * xp_powers[3] + a[5] * xp_powers[4];
        scalar_t Q = 1.0 + b0_abs * axp_powers[0] + b1_abs * axp_powers[1] + b2_abs * axp_powers[2] + b3_abs * axp_powers[3];
        scalar_t Q_inv = 1.0 / Q;
        scalar_t Q_inv2 = Q_inv * Q_inv;

        scalar_t grad_o = grad_output[index];
        scalar_t R = a[1] + 2.0 * a[2] * xp_powers[0] + 3.0 * a[3] * xp_powers[1] + 4.0 * a[4] * xp_powers[2] + 5.0 * a[5] * xp_powers[3];
        scalar_t S = copysign(1.0, xp) * (b0_abs + 2.0 * b1_abs * axp_powers[0] + 3.0 * b2_abs * axp_powers[1] + 4.0 * b3_abs * axp_powers[2]);

        scalar_t d_i_x = (R * Q_inv + S * (-P * Q_inv2)) * grad_o;
        d_x[index] = d_i_x;

        for (int i = 0; i < 6; ++i) {
            local_da[i] += xp_powers[i] * Q_inv * grad_o;
        }
        for (int i = 0; i < 4; ++i) {
            local_db[i] += (-P * Q_inv2) * copysign(1.0, b[i]) * axp_powers[i] * grad_o;
        }
    }

    // Reduce local arrays to shared memory
    for (int i = 0; i < 6; ++i) {
        atomicAdd(&sda[i], local_da[i]);
    }
    for (int i = 0; i < 4; ++i) {
        atomicAdd(&sdb[i], local_db[i]);
    }

    __syncthreads();

    // Only one thread writes back to global memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < 6; ++i) {
            atomicAdd(&d_a[i], sda[i]);
        }
        for (int i = 0; i < 4; ++i) {
            atomicAdd(&d_b[i], sdb[i]);
        }
    }
}

std::vector<torch::Tensor> rational_bwd_cuda_1dgroup(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = 512;  // You might want to experiment with this value
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rational_bwd_cuda_1dgroup", ([&] {
    rational_bwd_cuda_kernel_1dgroup<scalar_t>
        <<<numBlocks, blockSize>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<double>(),
            d_d.data_ptr<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}