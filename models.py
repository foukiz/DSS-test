import torch
import torch.nn as nn

from collections import OrderedDict

import opt_einsum as oe

from layers import DSSLayer, TopPooling, InputEncoder, Normalization
import layers




DEFAULT_STATE_SIZE = 64




class DSS(nn.Module):
    
    def __init__(
        self,
        input_size,
        output_size,
        data_dim,
        state_size=64,
        bidirectional=False,
        activation='gelu',
        kernel_version='exp',
        bias=True,
        dropout=0.0,
        normalization='batch_norm',
        n_layers=1,
        encoding=None,
        prenorm=False,
        residual=True,
        pooling='last',     # top pooling mode - 'last' or 'average' or 'manytomany'
        track_norms=False,
        seed=None,
        **kwargs
    ):
        assert n_layers > 0, (
            f"DSS model should have at least one core dss layer, found n_layers = {n_layers}")
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.activation = activation
        self.bias = bias
        self.version = kernel_version
        self.prenorm = prenorm
        self.residual = residual
        self.track_norms = track_norms
        if self.track_norms:
            self.layer_norms = {'layer_norm_{}'.format(i): 0. for i in range(n_layers)}
            self.layer_norms.update({'input_layer_norm': 0., 'pooling_layer_norm': 0.})

        self.input_layer = InputEncoder(data_dim, input_size, mode=encoding, **kwargs)
        self.normalization_layer = Normalization(input_size, mode=normalization)
        self.output_layer = nn.Linear(input_size, output_size, bias=bias)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.dss_blocks = []

        for i in range(n_layers):
            # stacker n_layers blocs DSS:
            # DSSLayer (core) + activation + dropout + linear (mixing layer)
            dss_block = nn.Sequential(OrderedDict([
                ('dss_layer', DSSLayer(input_size=input_size, state_size=state_size, version=self.version, bidirectional=bidirectional, bias=bias, **kwargs)),
                ('activation', self.activation),
                ('dropout', nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()),
                ('linear', nn.Linear(input_size, input_size, bias=bias))
            ]))
            setattr(self, f'dss_block_{i}', dss_block)
            self.dss_blocks.append(dss_block)

        # top pooling layer
        self.top_pooling = TopPooling(mode=pooling)

    def forward(self, u):
        x = self.input_layer(u)
        if self.track_norms: self.layer_norms['input_layer_norm'] += self.compute_layer_norm(x)
        for i, block in enumerate(self.dss_blocks):
            if self.residual: y = x
            if self.prenorm: x = self.normalization_layer(x)
            # DSS core computation + activation + dropout + linear mixing
            x = block(x)
            if self.residual: x = self.drop(x) + y
            if not self.prenorm: x = self.normalization_layer(x)
            if self.track_norms: self.layer_norms['layer_norm_{}'.format(i)] += self.compute_layer_norm(x)
        if self.prenorm: x = self.normalization_layer(x)
        x = self.top_pooling(x)
        if self.track_norms: self.layer_norms['pooling_layer_norm'] += self.compute_layer_norm(x, is_sequence=False)
        x = self.output_layer(x)
        if self.track_norms: self.layer_norms['output_norm'] += self.compute_layer_norm(x, is_sequence=False)
        return x

    def __str__(self):
        ret_str = str(self.input_layer) + "\n"
        ret_str += str(self.dss_block_0) + "\n"
        ret_str += "X {}".format(len(self.dss_blocks)) + "\n"
        ret_str += str(self.normalization_layer) + "\n"
        ret_str += str(self.top_pooling) + "\n"
        ret_str += str(self.output_layer)
        return ret_str

    def compute_norms(self, L):
        """ Compute the norms of the first item of the kernels and of the matrics D
            of each DSS layer, for monitoring purposes
        """
        
        norms = {}
        with torch.no_grad():
            for i, block in enumerate(self.dss_blocks):
                k = block.dss_layer.kernel(L)
                norms['norms/kernel_{}'.format(i)] = k[0].norm().item() / k[0].numel()
                norms['norms/D_{}'.format(i)] = block.dss_layer.D.norm().item() / block.dss_layer.D.numel()
        return norms

    def compute_layer_norm(self, x, is_sequence=True):
        """ Compute the norms of the first item of the sequence, averaged over batches
        """

        with torch.no_grad():
            # average along the batch dimension
            y = x.mean(dim=0)
            # keep only the first item of the sequence
            if is_sequence:
                y = y[0]

        return y.norm().item() / y.numel()
    
    def average_layer_norms(self, n_batches):
        for k in self.layer_norms.keys():
            self.layer_norms[k] /= n_batches

    def initialize_layer_norms(self):
        for k in self.layer_norms.keys():
            self.layer_norms[k] = 0.