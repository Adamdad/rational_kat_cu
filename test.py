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

if __name__=="__main__":
    degree = (5, 4)
    # Define tensors for the numerator and denominator coefficients
    numerator_weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    denominator_weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    # Input tensor
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    # Perform the rational function computation
    result = Rational_CUDA_A_F(x, numerator_weights, denominator_weights)

    # Print the result
    print("Result of Rational_CUDA_A_F:", result)
