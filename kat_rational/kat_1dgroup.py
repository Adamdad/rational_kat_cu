import torch
import kat_rational_cu
from torch import nn
import os
import json

class rational_1dgroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator, group):
        """
        Forward pass of the custom autograd function.
        
        Args:
            ctx: Context object used to stash information for backward computation.
            input (Tensor): Input tensor.
            weight_numerator (Tensor): Weights of the numerator polynomial.
            weight_denominator (Tensor): Weights of the denominator polynomial.
            group (int): The group number.

        Returns:
            Tensor: The result of the rational function applied to the input tensor.
        """
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        ctx.group = group
        x = kat_rational_cu.rational_fwd_1dgroup(input, weight_numerator, weight_denominator, group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the custom autograd function.
        
        Args:
            ctx: Context object from the forward pass.
            grad_output (Tensor): Gradient of the output tensor.

        Returns:
            tuple: Gradients of the input, weight_numerator, weight_denominator.
        """
        input, weight_numerator, weight_denominator = ctx.saved_tensors
        group = ctx.group
        d_input, d_weight_numerator, d_weight_denominator = kat_rational_cu.rational_bwd_1dgroup(grad_output, input, weight_numerator, weight_denominator, group)
        return d_input, d_weight_numerator, d_weight_denominator, None

class KAT_1DGroup(nn.Module):
    def __init__(self, num_groups=4, mode="searched"):
        """
        Initialize the KAT_1DGroup module.

        Args:
            num_groups (int): Number of groups to divide the input for separate processing.
            init_mode (str): Initialization mode which determines the preset weights from JSON file.
        """
        super(KAT_1DGroup, self).__init__()
        self.order = (5, 4)
        self.num_groups = num_groups
        # Initialize parameters for each group
        self.initialize(mode=mode)
        
    def initialize(self, mode="gelu"):
        """
        Initialize weights from a JSON file based on the specified mode.

        Args:
            mode (str): The initialization mode.
        """
        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"])
            weight_numerator = torch.cat([weight_numerator]*self.num_groups).view(self.num_groups, -1)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.cat([weight_denominator]*self.num_groups).view(self.num_groups, -1)
             
            self.weight_numerator = nn.Parameter(torch.FloatTensor(weight_numerator)
                                                      , requires_grad=True) 
            self.weight_denominator = nn.Parameter(torch.FloatTensor(weight_denominator)
                                                      , requires_grad=True) 
        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")
    
    def forward(self, input):
        """
        Forward pass of the module.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Processed tensor.
        """
        assert input.dim() == 3, "Input tensor must be 3D. Of size (batch, length, channels)."
        return rational_1dgroup.apply(input, self.weight_numerator, self.weight_denominator, self.num_groups)
    
    def extra_repr(self):
        """
        Extra representation of the module for debugging.

        Returns:
            str: String representation of the module's configuration.
        """
        return f'num_groups={self.num_groups}, order={self.order}'
    
class KAT_1DGroupv2(nn.Module):
    def __init__(self, num_groups=4, mode="searched"):
        """
        Initialize the KAT_1DGroup module.

        Args:
            num_groups (int): Number of groups to divide the input for separate processing.
            init_mode (str): Initialization mode which determines the preset weights from JSON file.
        """
        super(KAT_1DGroupv2, self).__init__()
        self.order = (5, 4)
        self.num_groups = num_groups
        # Initialize parameters for each group
        self.initialize(mode=mode)
        
    def initialize(self, mode="gelu"):
        """
        Initialize weights from a JSON file based on the specified mode.

        Args:
            mode (str): The initialization mode.
        """
        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"])
            weight_numerator = torch.cat([weight_numerator]*self.num_groups).view(self.num_groups, -1)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.cat([weight_denominator]*self.num_groups).view(self.num_groups, -1)
             
            self.weight_numerator = nn.Parameter(torch.FloatTensor(weight_numerator)
                                                      , requires_grad=True) 
            self.weight_denominator = nn.Parameter(torch.FloatTensor(weight_denominator)
                                                      , requires_grad=True) 

        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")
    
    def forward(self, input):
        """
        Forward pass of the module.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Processed tensor.
        """
        assert input.dim() == 3, "Input tensor must be 3D. Of size (batch, length, channels)."

        # select the first group, and repeat the weights for all groups
        weight_numerator = self.weight_numerator[0].repeat(self.num_groups, 1)
        weight_denominator = self.weight_denominator
        return rational_1dgroup.apply(input, weight_numerator, weight_denominator, self.num_groups)
    
    def extra_repr(self):
        """
        Extra representation of the module for debugging.

        Returns:
            str: String representation of the module's configuration.
        """
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
    
    