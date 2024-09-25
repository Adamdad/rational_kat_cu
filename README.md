# CUDA Rational Function for Kolmogorov–Arnold Transformer (KAT)

This CUDA C++ extension facilitates the use of group rational functions in Kolmogorov–Arnold Transformers (KAT). It support the training and inference of https://github.com/Adamdad/kat.

# Installation 
To install the extension, follow these steps:
```shell
git clone https://github.com/Adamdad/rational_kat_cu.git
cd rational_kat_cu
pip install -e .
```

# Usage
Incorporate the module into your neural network models as shown in the example below, which uses the rational function as an activation layer in a simple two-layer KAN architecture.
```python
from kat_rational import KAT_Group
class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode = act_cfg['act_init'][1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x
```

Note: The `KAT_Group` class accepts tensors with dimensions `[B, L, C]`, representing batch size, sequence length, and channel count, respectively. 

- [ ] We will try to implement the 2D version, with `[B, C, H, W]` soon. Stay tuned.

PS: Remember to `from kat_rational import KAT_Group` after `import torch`, to avoid errors.

PPS: I'm not a CUDA expert 😅. If you run into any issues or have suggestions for the code, please feel free to reach out or submit a pull request! 🚀

# Add new function 

To add new functions to the module:

1. Open `kat_rational/fit.py`.
2. Implement your custom function within this file.
3. Add your function to `fit_and_plot_activation` to evaluate and visualize its performance.

# Acknowlegement

We extend our gratitude to the [rational_activations](https://github.com/ml-research/rational_activations) project for providing the foundational CUDA implementation of rational functions.