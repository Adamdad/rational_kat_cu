import torch
from src_triton.rational_triton import rational_fwd_triton

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


# ======================================
# Testing both implementations for equality
# ======================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define dimensions and groups.
    B, L, D = 2, 4, 8  # Example: batch=2, length=4, features=8
    group = 4        # Make sure D is divisible by group
    assert D % group == 0, "D must be divisible by group"

    # Coefficient shapes.
    len_num = 6     # numerator coefficients (degree 5)
    len_deno = 4    # denominator coefficients (degree 3, plus constant)

    # Create random input and coefficients.
    x = torch.randn(B, L, D, device=device)
    weight_numerator = torch.randn(group, len_num, device=device)
    weight_denominator = torch.randn(group, len_deno, device=device)

    # Compute outputs.
    y_triton = rational_fwd_triton(x, weight_numerator, weight_denominator, group)
    y_torch  = Rational_CUDA_A_1DGroup(x, weight_numerator, weight_denominator, group)

    # Compute maximum absolute difference.
    diff = (y_triton - y_torch).abs().max().item()
    print("Maximum absolute difference between Triton and Torch implementations:", diff)

    # You can also assert closeness.
    if torch.allclose(y_triton, y_torch, atol=1e-5):
        print("SUCCESS: Triton and Torch implementations produce similar results.")
    else:
        print("WARNING: There is a difference between the implementations.")
    
    # Optionally, print the outputs.
    print("\nTriton output:")
    print(y_triton)
    print("\nTorch output:")
    print(y_torch)