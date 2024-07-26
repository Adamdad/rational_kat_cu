#include <torch/extension.h>

template <typename scalar_t>
__global__ void rational_fwd_cuda_kernel(
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, 
    scalar_t* __restrict__ result, 
    size_t x_size) {

    
    scalar_t a_0 = a[0];
    
    scalar_t a_1 = a[1];
    
    scalar_t a_2 = a[2];
    
    scalar_t a_3 = a[3];
    
    scalar_t a_4 = a[4];
    
    scalar_t a_5 = a[5];
    
    
    scalar_t ab_0 = abs(b[0]);
    
    scalar_t ab_1 = abs(b[1]);
    
    scalar_t ab_2 = abs(b[2]);
    
    scalar_t ab_3 = abs(b[3]);
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];
        scalar_t abs_xp1 = abs(xp1);

        // Horner's method for polynomial computation for P
        scalar_t P = a_5;
        P = P * xp1 + a_4;
        P = P * xp1 + a_3;
        P = P * xp1 + a_2;
        P = P * xp1 + a_1;
        P = P * xp1 + a_0;
        
        // Horner's method for polynomial computation for Q
        scalar_t Q = ab_3;
        Q = Q * abs_xp1 + ab_2;
        Q = Q * abs_xp1 + ab_1;
        Q = Q * abs_xp1 + ab_0;
        Q = Q * abs_xp1 + 1.0;

        result[index] = P / Q;
    }
}

torch::Tensor rational_fwd_cuda(
    torch::Tensor x, torch::Tensor n, torch::Tensor d
    ){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();
    int blockSize = 512;  // You might want to experiment with this value
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rational_fwd_cuda", ([&] {
    rational_fwd_cuda_kernel<scalar_t>
        <<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x_size);
        }));

    return result;
}