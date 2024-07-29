import torch
import kat_rational
from torch import nn
import os
import json


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
    
class KAT_1DGroup(nn.Module):
    def __init__(self, num_groups=4, init_mode="gelu"):
        super(KAT_1DGroup, self).__init__()
        self.order = (5, 4)
        self.num_groups = num_groups
        # Initialize parameters for each group
        self.weight_numerator = nn.Parameter(torch.randn(num_groups, self.order[0]+1), requires_grad=True)
        self.weight_denominator = nn.Parameter(torch.randn(num_groups, self.order[1]), requires_grad=True)
        self.initialize(mode = init_mode)
        
    def initialize(self, mode="gelu"):
        cfd = os.path.dirname(os.path.realpath(__file__))
        with open(f'{cfd}/init.json') as json_file:
            data = json.load(json_file)
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"])
            weight_numerator = torch.cat([weight_numerator]*self.num_groups).view(self.num_groups, -1)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.cat([weight_denominator]*self.num_groups).view(self.num_groups, -1)
        
        
        self.weight_numerator.data = weight_numerator
        self.weight_denominator.data = weight_denominator

    def forward(self, input):
        return My_rational_1dgroup.apply(input, self.weight_numerator, self.weight_denominator, self.num_groups)
    
    def extra_repr(self):
        return f'num_groups={self.num_groups}, order={self.order}'
    
if __name__=="__main__":
    model = KAT_1DGroup()
    x = torch.linspace(-1, 1, 100)
    # y = model(x)
    y = Rational_CUDA_A_F(x, model.weight_numerator[0], model.weight_denominator[0])
    # plot y vs x
    import matplotlib.pyplot as plt
    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.savefig("kat_1dgroup.png")

    
    