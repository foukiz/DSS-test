import torch
import torch.nn as nn

import opt_einsum as oe

from layers import DSSLayer, TopPooling, InputEncoder, Normalization






class DSS(nn.Module):
    
    def __init__(
        self,
        input_size,
        state_size,
        output_size,
        data_dim,
        bidirectional=False,
        activation='gelu',
        kernel_version='exp',
        bias=True,
        dropout=0.0,
        normalization='batch_norm',
        n_layers=1,
        encoding=None,
        pooling='last',     # top pooling mode - 'last' or 'average' or 'manytomany'
        seed=None
    ):
        if seed: torch.manual_seed(seed)
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.activation = activation
        self.bias = bias
        self.version = kernel_version

        self.input_layer = InputEncoder(data_dim, input_size, mode=encoding)
        self.normalization_layer = Normalization(input_size, mode=normalization)
        self.output_layer = nn.Linear(input_size, output_size, bias=bias)

        layers = []
        for _ in range(n_layers):
            # stacker n_layers blocs DSS:
            # DSSLayer (core) + activation + dropout + linear (mixing layer)
            layers.append(
                DSSLayer(input_size=input_size, state_size=state_size, version=self.version, bidirectional=bidirectional, bias=bias)
            )
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
            layers.append(nn.Linear(input_size, input_size, bias=bias))

        self.dss_layers = nn.ModuleList(layers)

        # top pooling layer
        self.top_pooling = TopPooling(mode=pooling)

    def forward(self, u):
        x = self.input_layer(u)
        for layer in self.dss_layers:
            x = layer(x)
        x = self.normalization_layer(x)
        x = self.top_pooling(x)
        x = self.output_layer(x)
        return x
