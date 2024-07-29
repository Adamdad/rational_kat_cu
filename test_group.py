import torch
import kat_rational
from rational.torch import Rational
from torch import nn

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


def Rational_CUDA_A_1DGroup(x, weight_numerator, weight_denominator, group):
    """
    Computes the rational function P(x) / Q(x) group-wise where P and Q are polynomials defined by
    the given weights for their coefficients for each group.
    P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
                1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|
    
    Args:
    - x (torch.Tensor): Input tensor of shape (B, L, D).
    - weight_numerator (torch.Tensor): Coefficients of the numerator polynomial for each group.
                                       Shape (group, len_num).
    - weight_denominator (torch.Tensor): Coefficients of the denominator polynomial for each group.
                                         Shape (group, len_deno).
    
    Returns:
    - torch.Tensor: Result of the rational function computation of shape (B, L, D).
    """
    device = x.device
    B, L, D = x.shape
    len_num = weight_numerator.size(1)
    len_deno = weight_denominator.size(1)

    # Group-wise application, ensure D is divisible by the number of groups
    D_per_group = D // group

    # Reshape x to apply each group's parameters separately
    z = x.view(B, L, group, D_per_group).permute(2, 0, 1, 3).contiguous()  # Shape: (group, B, L, D_per_group)
    z = z.view(group, B * L * D_per_group)  # Flatten for group-wise operation

    # Generate powers of z for polynomial terms, assuming _get_xps function supports batched operation
    xps = _get_xps(z, len_num, len_deno)  # Should output shape: (group, B * L * D_per_group, max(len_num, len_deno))

    # Compute numerator as a dot product of powers of z and weights
    numerator = torch.bmm(weight_numerator.unsqueeze(1), xps).squeeze(1)  # Shape: (group, B * L * D_per_group)

    # Compute denominator similarly, considering absolute values
    expanded_dw = torch.cat([
        torch.ones(group, 1, device=device),  # 1 for the constant term of denominator
        weight_denominator,
        torch.zeros(group, max(0, len_num - len_deno - 1), device=device)  # Pad with zeros if numerator degree is higher
    ], dim=1)

    denominator = torch.bmm(expanded_dw.abs().unsqueeze(1), xps).squeeze(1)  # Shape: (group, B * L * D_per_group)

    # Compute the rational function result
    result = numerator.div(denominator)

    # Reshape and reorder to match original x shape
    result = result.view(group, B, L, D_per_group).permute(1, 2, 0, 3).contiguous()  # Shape: (B, L, group, D_per_group)
    result = result.view(B, L, D)  # Shape: (B, L, D)

    return result

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

def process_groups(B, L, D, group, x, weights_numerator, weights_denominator):
    """
    Applies Rational_CUDA_A_F group-wise to an input tensor of shape (B, L, D).
    
    Args:
    - B, L, D (int): Dimensions of the input tensor.
    - group (int): Number of groups.
    - x (torch.Tensor): Input tensor of shape (B, L, D).
    - weights_numerator (list of torch.Tensor): List of tensors, each containing numerator coefficients for a group.
    - weights_denominator (list of torch.Tensor): List of tensors, each containing denominator coefficients for a group.
    
    Returns:
    - torch.Tensor: The result tensor of shape (B, L, D).
    """
    
    D_per_group = D // group
    results = []

    for g in range(group):
        # Slice the input tensor for the current group
        start_idx = g * D_per_group
        end_idx = start_idx + D_per_group
        x_group = x[:, :, start_idx:end_idx]
        x_group = x_group.contiguous()

        # Compute the rational function for the current group
        result_group = Rational_CUDA_A_F(x_group, weights_numerator[g], weights_denominator[g])
        results.append(result_group)

    # Concatenate the results along the depth dimension
    return torch.cat(results, dim=2)

class My_rational_1dgroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator, group):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        ctx.group = group
        x = kat_rational.rational_fwd_1dgroup(input, weight_numerator, weight_denominator, group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, w_numerator, w_denominator = ctx.saved_tensors
        group = ctx.group
        d_x, d_weight_numerator, d_weight_denominator = kat_rational.rational_bwd_1dgroup(grad_output, x, w_numerator, w_denominator, group)
        return d_x, d_weight_numerator, d_weight_denominator, None

def test_vectorized_forward(x, numerator_weights, denominator_weights, group_size=4):
        
        print("Testing forward pass")
        B, L, D = x.shape
        # Perform the rational function computation
        loop_results = process_groups(B, L, D, group_size, x, numerator_weights, denominator_weights)
        vector_result = Rational_CUDA_A_1DGroup(x, numerator_weights, denominator_weights, group_size)
    
        # Check if the results match
        assert torch.allclose(loop_results, vector_result)
        print("Forward pass test passed")

def test_forward(x, numerator_weights, denominator_weights, group_size=4):
    
    print("Testing forward pass")
    # Perform the rational function computation
    # loop_results = process_groups(1024, 77, 640, group_size, x, numerator_weights, denominator_weights)
    
    vector_result = Rational_CUDA_A_1DGroup(x, numerator_weights, denominator_weights, group_size)

    my_results = My_rational_1dgroup.apply(x, numerator_weights, denominator_weights, group_size)
    print("My results shape:", my_results.shape)
    print(my_results[0])
    
    print("Vectorized results shape:", vector_result.shape)
    print(vector_result[0])
    
    assert torch.allclose(vector_result[0], my_results[0]), "First element mismatch"
    assert torch.allclose(vector_result[:, 1], my_results[:,1]), "Second element mismatch"
    assert torch.allclose(vector_result[:, :, 0], my_results[:,:, 0]), "Third element mismatch"
    # Check if the results match
    assert torch.allclose(vector_result, my_results)
    print("Forward pass test passed")
    print("#"*50)

def test_backward(x, numerator_weights, denominator_weights, group_size=4):
    print("Testing backward pass")
    expected_output = torch.sigmoid(x)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Perform the rational function computation
    output = Rational_CUDA_A_1DGroup(x, numerator_weights, denominator_weights, group_size)
    print("torch output", output)
    loss = loss_fn(expected_output, output)
    loss.backward()
    torch_grad_n = numerator_weights.grad
    torch_grad_d = denominator_weights.grad
    
    numerator_weights.grad.zero_()
    denominator_weights.grad.zero_()
    
    my_output = My_rational_1dgroup.apply(x, numerator_weights, denominator_weights, group_size)
    print("my output", my_output)
    loss = loss_fn(expected_output, my_output)
    loss.backward()
    my_grad_n = numerator_weights.grad
    my_grad_d = denominator_weights.grad
    
    print("Torch grad numerator:", torch_grad_n)
    print("My grad numerator:", my_grad_n)
    print("Torch grad denominator:", torch_grad_d)
    print("My grad denominator:", my_grad_d)
    
    assert torch.allclose(torch_grad_n, my_grad_n), "Numerator gradient mismatch"
    assert torch.allclose(torch_grad_d, my_grad_d), "Denominator gradient mismatch"
    
    print("Backward pass test passed")

def benchmark_forward(x, numerator_weights, denominator_weights, group_size=4):
    import time
    print("Benchmarking forward pass")
    B, L, D = x.shape
    used_time = 0
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    for _ in range(100):
        start = time.time()
        result = process_groups(B, L, D, group_size, x, numerator_weights, denominator_weights)
        torch.cuda.synchronize()
        used_time += time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    used_time /= 100
    print("Time taken for loop forward pass: {:.4f} seconds".format(used_time), "Peak memory:", peak_mem)
    
    used_time = 0
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    for _ in range(100):
        start = time.time()
        result = Rational_CUDA_A_1DGroup(x, numerator_weights, denominator_weights, group_size)
        torch.cuda.synchronize()
        used_time += time.time() - start

    used_time /= 100
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    print("Time taken for torch vectorized forward pass: {:.4f} seconds".format(used_time), "Peak memory:", peak_mem)
    
    
    used_time = 0
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    for _ in range(100):
        start = time.time()
        result = My_rational_1dgroup.apply(x, numerator_weights, denominator_weights, group_size)
        torch.cuda.synchronize()
        used_time += time.time() - start

    used_time /= 100
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    print("Time taken for cuda forward pass: {:.4f} seconds".format(used_time), "Peak memory:", peak_mem)
    
    print("#"*50)
    return result

if __name__=="__main__":
    group_size = 4
    # Define tensors for the numerator and denominator coefficients
    # numerator of size (group_size, 5) and denominator of size (group_size, 4)
    numerator_weights = nn.Parameter(torch.randn(4, 6, dtype=torch.float32, device='cuda')/640, requires_grad=True)
    denominator_weights = nn.Parameter(torch.randn(4, 4, dtype=torch.float32, device='cuda')/640, requires_grad=True)
    # numerator_weights = nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32, device='cuda'), requires_grad=True)
    # denominator_weights = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device='cuda'), requires_grad=True)

    # Input tensor
    x = torch.rand(512, 10, 32, dtype=torch.float32, device='cuda')
    # test_forward(x, numerator_weights, denominator_weights, group_size)
    # benchmark_forward(x, numerator_weights, denominator_weights, group_size)
    test_backward(x, numerator_weights, denominator_weights, group_size)
    
    
    
    
    