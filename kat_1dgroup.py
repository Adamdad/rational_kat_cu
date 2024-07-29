import torch
import kat_rational
from torch import nn
import os
import json

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
        cfd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(f'{cfd}/init.json') as json_file:
            data = json.load(json_file)
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"])
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
        
        self.weight_numerator.data = weight_numerator
        self.weight_denominator.data = weight_denominator

    def forward(self, input):
        return My_rational_1dgroup.apply(input, self.weight_numerator, self.weight_denominator, self.num_groups)
    
    def extra_repr(self):
        return f'num_groups={self.num_groups}, order={self.order}'
    
if __name__=="__main__":
    model = KAT_1DGroup()
    x = torch.linspace(-1, 1, 100).view(1, 1, -1)
    y = model(x)
    # plot y vs x
    import matplotlib.pyplot as plt
    plt.plot(x[0, 0, :].numpy(), y[0, 0, :].detach().numpy())
    plt.savefig("kat_1dgroup.png")

    
    