import torch
import my_lib

def _get_xps(z, len_numerator, len_denominator):
    """
    Generates a tensor of powers of the input tensor `z` up to the maximum order 
    needed for the numerator or denominator, whichever is higher.
    
    Args:
    - z (torch.Tensor): The input tensor for which powers are computed.
    - len_numerator (int): Degree of the numerator polynomial plus one.
    - len_denominator (int): Degree of the denominator polynomial plus one.
    
    Returns:
    - torch.Tensor: Tensor where each row contains powers of `z` from 0 to max degree.
    """
    xps = [z]
    for _ in range(max(len_numerator, len_denominator) - 2):
        xps.append(xps[-1] * z)
    xps.insert(0, torch.ones_like(z))  # Add x^0 = 1
    return torch.stack(xps, dim=1)


def Rational_CUDA_A_F(x, weight_numerator, weight_denominator):
    """
    Computes the rational function P(x) / Q(x) where P and Q are polynomials defined by
    the given weights for their coefficients.
    P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
                1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|
    
    Args:
    - x (torch.Tensor): Input tensor.
    - weight_numerator (torch.Tensor): Coefficients of the numerator polynomial.
    - weight_denominator (torch.Tensor): Coefficients of the denominator polynomial.
    
    Returns:
    - torch.Tensor: Result of the rational function computation.
    """
    device = weight_numerator.device
    z = x.view(-1)  # Flatten x to a 1D tensor
    len_num, len_deno = len(weight_numerator), len(weight_denominator)

    # Generate powers of z for polynomial terms
    xps = _get_xps(z, len_num, len_deno)

    # Compute the numerator as a dot product of powers of z and weights
    numerator = (xps * weight_numerator).sum(dim=1)

    # Prepare denominator weights with zero-padding as necessary
    expanded_dw = torch.cat([
        torch.tensor([1.]).to(device),  # 1 for the constant term of denominator
        weight_denominator,
        torch.zeros(max(0, len_num - len_deno - 1)).to(device)  # Pad with zeros if numerator degree is higher
    ])

    # Compute the denominator similarly, considering absolute values
    denominator = (xps * expanded_dw).abs().sum(dim=1)

    return numerator.div(denominator).view(x.shape)  # Reshape result to match input shape

class My_rational(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        my_lib.rational_fwd(input, weight_numerator, weight_denominator)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():  # TODO this check is necessary if efficientnet is used
            print("grad_output is not contiguous")
            grad_output = grad_output.contiguous()
        x, w_numerator, w_denominator = ctx.saved_tensors
        print(grad_output.shape, grad_output.dtype)
        d_x, d_weight_numerator, d_weight_denominator = my_lib.rational_bwd(grad_output, x, w_numerator, w_denominator)
        return d_x, d_weight_numerator, d_weight_denominator, None

def test_forward(x, numerator_weights, denominator_weights):
    # Perform the rational function computation
    result = Rational_CUDA_A_F(x, numerator_weights, denominator_weights)

    my_results = My_rational.apply(x, numerator_weights, denominator_weights)

    # Check if the results match
    assert torch.allclose(result, my_results)

    return result

def test_backward(x, numerator_weights, denominator_weights):
    # Perform the rational function computation
    # result = Rational_CUDA_A_F(x, numerator_weights, denominator_weights)
    # result.sum().backward()
    # torch_grad = x.grad
    
    # for p in [x, numerator_weights, denominator_weights]:
    #     p.grad.detach_()
    #     p.grad.zero_()

    my_results = My_rational.apply(x, numerator_weights, denominator_weights)
    my_results.sum().backward()
    
    my_grad = x.grad

    # Check if the results match
    # assert torch.allclose(torch_grad, my_grad)

    # return result

def benchmark_time(x, numerator_weights, denominator_weights):
    import time
    used_time = 0
    for _ in range(100):
        start = time.time()
        result = Rational_CUDA_A_F(x, numerator_weights, denominator_weights)
        torch.cuda.synchronize()
        used_time += time.time() - start

    used_time /= 100
    print("Time taken by Rational_CUDA_A_F:", used_time)

    used_time = 0
    for _ in range(100):
        start = time.time()
        my_results = my_lib.rational_fwd(x, numerator_weights, denominator_weights)
        torch.cuda.synchronize()
        used_time += time.time() - start

    used_time /= 100
    print("Time taken by my_lib.rational_fwd:", used_time)

    return result
if __name__=="__main__":
    # Define tensors for the numerator and denominator coefficients
    numerator_weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32, device='cuda', requires_grad=True)
    denominator_weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device='cuda', requires_grad=True)

    # Input tensor
    x = torch.randn(100, 100, dtype=torch.float32, device='cuda', requires_grad=True)
    test_forward(x, numerator_weights, denominator_weights)
    # test_backward(x, numerator_weights, denominator_weights)