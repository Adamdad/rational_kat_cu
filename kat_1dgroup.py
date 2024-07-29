import torch
import kat_rational
from torch import nn
import os
import json

class rational_1dgroup(torch.autograd.Function):
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
    
class rational(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        x = kat_rational.rational_fwd_optimized(input, weight_numerator, weight_denominator)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, w_numerator, w_denominator = ctx.saved_tensors
        d_x, d_weight_numerator, d_weight_denominator = kat_rational.rational_bwd_optimized(grad_output, x, w_numerator, w_denominator)
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
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)
            # Initialize weights
        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")
        weight_numerator = torch.tensor(data[mode]["init_w_numerator"])
        weight_numerator = torch.cat([weight_numerator]*self.num_groups).view(self.num_groups, -1)
        weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
        weight_denominator = torch.cat([weight_denominator]*self.num_groups).view(self.num_groups, -1)
        
        self.weight_numerator.data = weight_numerator
        self.weight_denominator.data = weight_denominator

    def forward(self, input):
        return rational_1dgroup.apply(input, self.weight_numerator, self.weight_denominator, self.num_groups)
    
    def extra_repr(self):
        return f'num_groups={self.num_groups}, order={self.order}'

    
if __name__=="__main__":
    
    model = KAT_1DGroup()
    x = torch.linspace(-2, 2, 100)
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    
    x = x.unsqueeze(0).unsqueeze(0)
    y = model(x.cuda())
    x = x.squeeze(0).squeeze(0)
    y = y.squeeze(0).squeeze(0)
    # plot y vs x
    import matplotlib.pyplot as plt
    plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Response of KAT_1DGroup")
    plt.grid(True)
    plt.savefig("kat_1dgroup.png")
    
    