import torch
from src_triton.rational_triton import rational_fwd as triton_rational_fwd
from kat_rational.kat_1dgroup_torch import Rational_CUDA_A_1DGroup
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
    y_triton = triton_rational_fwd(x, weight_numerator, weight_denominator, group)
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