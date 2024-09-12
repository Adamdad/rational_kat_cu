import torch
from kat_rational import rational_1dgroup
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
    
    act = Rational(approx_func="gelu").cuda()
    act.numerator.data = numerator_weights[0]
    act.denominator.data = denominator_weights[0]
    B, L, D = x.shape
    print("Testing forward pass")
    # Perform the rational function computation
    vector_result = process_groups(B, L, D, group_size, x, numerator_weights, denominator_weights)
    
    # vector_result = Rational_CUDA_A_1DGroup(x, numerator_weights, denominator_weights, group_size)
    
    rational_output = act(x)

    my_results = rational_1dgroup.apply(x, numerator_weights, denominator_weights, group_size)

    # print(rational_output)
    # print(vector_result)
    # print(my_results)
    print("rational_output - vector_result", rational_output - vector_result)
    print("my_results - vector_result", my_results - vector_result)
    assert torch.allclose(rational_output, my_results, atol=1e-07), "Output mismatch"    
    # Check if the results match
    assert torch.allclose(vector_result, my_results, atol=1e-07)
    print("Forward pass test passed")
    print("#"*50)
    

def test_backward(x, numerator_weights, denominator_weights, group_size=4):
    print("Testing backward pass")
    B, L, D = x.shape
    expected_output = torch.cat([torch.sigmoid(x[:,:,:D//2]), torch.relu(x[:,:,D//2:])], dim=2)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    # Perform the rational function computation
    output = process_groups(B, L, D, group_size, x, numerator_weights, denominator_weights)
    # output = Rational_CUDA_A_1DGroup(x, numerator_weights, denominator_weights, group_size)
    loss = loss_fn(expected_output, output)
    loss.backward()
    torch_grad_n = numerator_weights.grad.clone()
    torch_grad_d = denominator_weights.grad.clone()
    
    numerator_weights.grad.zero_()
    denominator_weights.grad.zero_()
    
    my_output = rational_1dgroup.apply(x, numerator_weights, denominator_weights, group_size)
    loss = loss_fn(expected_output, my_output)
    loss.backward()
    my_grad_n = numerator_weights.grad.clone()
    my_grad_d = denominator_weights.grad.clone()
    
    print(output-my_output )
    
    print(torch_grad_n)
    print(my_grad_n)
    print(torch_grad_d)
    print(my_grad_d)
    assert torch.allclose(my_output, output, atol=1e-07), "Output mismatch"
    assert torch.allclose(torch_grad_n, my_grad_n, atol=1e-07), "Numerator gradient mismatch"
    assert torch.allclose(torch_grad_d, my_grad_d, atol=1e-07), "Denominator gradient mismatch"
    
    print("Backward pass test passed")
    
    # print(numerator_weights, denominator_weights)

def benchmark_backward(x, numerator_weights, denominator_weights, group_size=4):
    import time
    B, L, D = x.shape
    # expected_output = torch.sigmoid(x)
    expected_output = torch.cat([torch.sigmoid(x[:,:,:D//2]), torch.relu(x[:,:,D//2:])], dim=2)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    used_time = 0
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    start = time.time()
    for _ in range(100):
        output = process_groups(B, L, D, group_size, x, numerator_weights, denominator_weights)
        loss = loss_fn(expected_output, output)
        loss.backward()
        torch.cuda.synchronize()
        numerator_weights.grad.detach_()
        numerator_weights.grad.zero_()
        denominator_weights.grad.detach_()
        denominator_weights.grad.zero_()
        
    used_time += time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    
    used_time /= 100
    print("Time taken by loop bwd:", used_time, "Peak memory:", peak_mem)
    
    used_time = 0
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    start = time.time()
    for _ in range(100):
        my_output = Rational_CUDA_A_1DGroup(x, numerator_weights, denominator_weights, group_size)
        loss = loss_fn(expected_output, my_output)
        loss.backward()
        torch.cuda.synchronize()
        
        numerator_weights.grad.detach_()
        numerator_weights.grad.zero_()
        denominator_weights.grad.detach_()
        denominator_weights.grad.zero_()
    used_time += time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    used_time /= 100
    print("Time taken by torch vectorized bwd:", used_time, "Peak memory:", peak_mem)
    
    used_time = 0
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    start = time.time()
    for _ in range(100):
        my_output = rational_1dgroup.apply(x, numerator_weights, denominator_weights, group_size)
        loss = loss_fn(expected_output, my_output)
        loss.backward()
        torch.cuda.synchronize()
        numerator_weights.grad.detach_()
        numerator_weights.grad.zero_()
        denominator_weights.grad.detach_()
        denominator_weights.grad.zero_()
        
    used_time += time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
        
    used_time /= 100
    print("Time taken by kat_rational.My_rational_1dgroup bwd:", used_time, "Peak memory:", peak_mem)

def benchmark_forward(x, numerator_weights, denominator_weights, group_size=4):
    import time
    """
    Benchmarking the forward pass of different methods for rational function computation.
    
    Args:
        x (Tensor): Input tensor of shape (B, L, D).
        numerator_weights (Tensor): Weights for the numerator polynomial.
        denominator_weights (Tensor): Weights for the denominator polynomial.
        group_size (int): Group size for rational function.
        
    Returns:
        dict: Results containing throughput and peak memory for each method.
    """
    
    B, L, D = x.shape
    num_batches = 1000  # Number of iterations to run for benchmarking
    
    print("Benchmarking forward pass")
    
    results = {}

    # Method 1: Loop-based processing groups
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    start_time = time.time()
    for _ in range(num_batches):
        result = process_groups(B, L, D, group_size, x, numerator_weights, denominator_weights)
        torch.cuda.synchronize()  # Ensure CUDA kernel completion
    total_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    throughput = num_batches / total_time
    print(f"Throughput for loop forward pass: {throughput:.2f} batches/sec, Peak memory: {peak_mem:.2f} MB")
    results['loop_forward_pass'] = {'throughput': throughput, 'peak_memory': peak_mem}

    # Method 2: Rational_CUDA_A_1DGroup (vectorized CUDA)
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    start_time = time.time()
    for _ in range(num_batches):
        result = Rational_CUDA_A_1DGroup(x, numerator_weights, denominator_weights, group_size)
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    throughput = num_batches / total_time
    print(f"Throughput for torch vectorized forward pass: {throughput:.2f} batches/sec, Peak memory: {peak_mem:.2f} MB")
    results['vectorized_forward_pass'] = {'throughput': throughput, 'peak_memory': peak_mem}

    # Method 3: rational_1dgroup.apply (CUDA autograd custom function)
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    start_time = time.time()
    for _ in range(num_batches):
        result = rational_1dgroup.apply(x, numerator_weights, denominator_weights, group_size)
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    throughput = num_batches / total_time
    print(f"Throughput for cuda forward pass: {throughput:.2f} batches/sec, Peak memory: {peak_mem:.2f} MB")
    results['cuda_forward_pass'] = {'throughput': throughput, 'peak_memory': peak_mem}
    
    # # Method 4: GELU activation function
    # start_time = time.time()
    # for _ in range(num_batches):
    #     result = torch.nn.functional.gelu(x)
    #     torch.cuda.synchronize()
    # total_time = time.time() - start_time
    # peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    # throughput = num_batches / total_time
    # print(f"Throughput for GELU forward pass: {throughput:.2f} batches/sec, Peak memory: {peak_mem:.2f} MB")
    # results['gelu_forward_pass'] = {'throughput': throughput, 'peak_memory': peak_mem}
    
    # # Method 5: ReLU activation function
    # start_time = time.time()
    # for _ in range(num_batches):
    #     result = torch.nn.functional.relu(x)
    #     torch.cuda.synchronize()
    # total_time = time.time() - start_time
    # peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    # throughput = num_batches / total_time
    # print(f"Throughput for ReLU forward pass: {throughput:.2f} batches/sec, Peak memory: {peak_mem:.2f} MB")
    # results['relu_forward_pass'] = {'throughput': throughput, 'peak_memory': peak_mem}
    
    # # Method 5: Silu activation function
    # start_time = time.time()
    # for _ in range(num_batches):
    #     result = torch.nn.functional.silu(x)
    #     torch.cuda.synchronize()
    # total_time = time.time() - start_time
    # peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    # throughput = num_batches / total_time
    # print(f"Throughput for silu forward pass: {throughput:.2f} batches/sec, Peak memory: {peak_mem:.2f} MB")
    # results['silu_forward_pass'] = {'throughput': throughput, 'peak_memory': peak_mem}
    
    # # Method 6: Sigmoid activation function
    # start_time = time.time()
    # for _ in range(num_batches):
    #     result = torch.nn.functional.sigmoid(x)
    #     torch.cuda.synchronize()
    # total_time = time.time() - start_time
    # peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    # throughput = num_batches / total_time
    # print(f"Throughput for sigmoid forward pass: {throughput:.2f} batches/sec, Peak memory: {peak_mem:.2f} MB")
    # results['sigmoid_forward_pass'] = {'throughput': throughput, 'peak_memory': peak_mem}
    
    # act = torch.nn.PReLU().cuda()
    # # Method 7: PReLU activation function
    # start_time = time.time()
    # for _ in range(num_batches):
    #     result = act(x)
    #     torch.cuda.synchronize()
    # total_time = time.time() - start_time
    # peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    # throughput = num_batches / total_time
    # print(f"Throughput for prelu forward pass: {throughput:.2f} batches/sec, Peak memory: {peak_mem:.2f} MB")
    # results['prelu_forward_pass'] = {'throughput': throughput, 'peak_memory': peak_mem}
    
    
    print("#" * 50)
    
    return results

if __name__=="__main__":
    for D in [128, 256, 512, 1024, 2048]:
        # print(f"Group size: {group_size}")
        print(f"Input shape: (64, 1000, {D})")
        group_size = 8
    # group_size = 8
        # Define tensors for the numerator and denominator coefficients
        # numerator of size (group_size, 5) and denominator of size (group_size, 4)
        numerator_weights = nn.Parameter(torch.tensor([
            [
                    -0.0012423594497499122,
                    0.5080497063245629,
                    0.41586363182937475,
                    0.13022718688035761,
                    0.024355900098993424,
                    0.00290283948155535
                ]]*group_size, dtype=torch.float32, device='cuda'), requires_grad=True)
        denominator_weights = nn.Parameter(torch.tensor([[
                    -0.06675015696494944,
                    0.17927646217001553,
                    0.03746682605496631,
                    1.6561610853276082e-10
                ]] * group_size, dtype=torch.float32, device='cuda'), requires_grad=True)
        # numerator_weights.data[1] *= 2
        # numerator_weights.data[2] *= 3
        # numerator_weights.data[3] *= 4

        # Input tensor
        x = torch.randn(64, 1000, D, dtype=torch.float32, device='cuda')
        # test_forward(x, numerator_weights, denominator_weights, group_size)
        benchmark_forward(x, numerator_weights, denominator_weights, group_size)
    # test_backward(x, numerator_weights, denominator_weights, group_size)
    # benchmark_backward(x, numerator_weights, denominator_weights, group_size)
    
    
    
    