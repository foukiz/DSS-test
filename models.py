import torch
import torch.nn as nn

from collections import OrderedDict

import opt_einsum as oe

from layers import DSSLayer, S4Layer, TopPooling, InputEncoder, Normalization, RetrievalHead
from transformer_layers import EncoderLayer, DecoderLayer, PositionalEncoding



DEFAULT_STATE_SIZE = 64




class DSS(nn.Module):
    
    def __init__(
        self,
        input_size,
        output_size,
        data_dim,
        state_size=DEFAULT_STATE_SIZE,
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
        retrieval=False,
        use_lengths=False,
        track_norms=False,
        seed=None,
        **kwargs
    ):
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
        self.retrieval = retrieval
        if self.track_norms:
            self.layer_norms = {'layer_norm_{}'.format(i): 0. for i in range(n_layers)}
            self.layer_norms.update({'input_layer_norm': 0., 'pooling_layer_norm': 0., 'output_norm': 0.})

        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.dss_blocks = []
        self.inner_normalizations = []

        for i in range(n_layers):
            # stacker n_layers blocs DSS:
            # DSSLayer (core) + activation + dropout + linear (mixing layer)
            dss_layer = DSSLayer(input_size=input_size, state_size=state_size, activation=activation, dropout=dropout, version=self.version, bidirectional=bidirectional, bias=bias, **kwargs)
            setattr(self, f'core_block_{i}', dss_layer)
            self.dss_blocks.append(dss_layer)
            setattr(self, f"inner_normalization_{i}", Normalization(input_size, mode=normalization))
            self.inner_normalizations.append(getattr(self, f"inner_normalization_{i}"))

        self.post_norm = Normalization(input_size, mode=normalization) if prenorm else nn.Identity()
        self.input_layer = InputEncoder(data_dim, input_size, mode=encoding, **kwargs)
        self.output_layer = nn.Linear(input_size, output_size, bias=bias)
        # top pooling layer
        self.top_pooling = TopPooling(mode=pooling, use_lengths=use_lengths)
        if self.retrieval:
            self.retrieval_head = RetrievalHead(input_size, output_size, activation='gelu')

    def forward(self, u, batch_lengths=None, transpose=True):
        """ Input u should be of shape (B, L) if encoding is 'embedding', else (B, L, data_dim)"""
        x = self.input_layer(u)
        if transpose: x = x.transpose(-1, -2)  # (B H L)
        if self.track_norms: self.layer_norms['input_layer_norm'] += self.compute_layer_norm(x, transpose=transpose)
        for i, layer in enumerate(self.dss_blocks):
            if self.residual: y = x
            if self.prenorm: x = self.inner_normalizations[i](x)
            # DSS core computation + activation + dropout + linear mixing
            x = layer(x)  # (B L H) / (B H L)
            if self.residual: x = self.drop(x) + y
            if not self.prenorm: x = self.inner_normalizations[i](x)
            if self.track_norms: self.layer_norms['layer_norm_{}'.format(i)] += self.compute_layer_norm(x, transpose=transpose)
        # if prenorm, just the identity
        x = self.post_norm(x)
        # pooling along the length dimension
        if transpose: x = x.transpose(-1, -2)  # (B H L) -> (B L H)
        x = self.top_pooling(x, batch_lengths=batch_lengths)
        if self.track_norms: self.layer_norms['pooling_layer_norm'] += self.compute_layer_norm(x, is_sequence=False, transpose=transpose)
        # if the model is applied to the retrieval task, use the retrieval head
        # as the output layer, instead of the standard output layer
        if self.retrieval:
            x = self.retrieval_head(x)
            if self.track_norms: self.layer_norms['output_norm'] += self.compute_layer_norm(x, is_sequence=False, transpose=transpose)
            return x
        x = self.output_layer(x)
        if self.track_norms: self.layer_norms['output_norm'] += self.compute_layer_norm(x, is_sequence=False, transpose=transpose)
        return x

    def __str__(self):
        ret_str = str(self.input_layer) + "\n"
        ret_str += str(self.core_block_0) + "\n"
        ret_str += "X {}".format(len(self.dss_blocks)) + "\n"
        ret_str += str(self.post_norm) + "\n"
        ret_str += str(self.top_pooling) + "\n"
        ret_str += str(self.output_layer)
        return ret_str

    def compute_norms(self):
        """ Compute the norms of the first item of the kernels and of the matrics D
            of each DSS layer, for monitoring purposes
        """

        norms = {}
        with torch.no_grad():
            for i, block in enumerate(self.dss_blocks):
                k = block.kernel
                #norms['norms/kernel_{}'.format(i)] = k[0].norm().item() / k[0].numel()
                kernel_norms = k.compute_norms()
                norms.update(
                    {k + '_{}'.format(i): v for k, v in kernel_norms.items()}
                )
                norms['norms/D_{}'.format(i)] = block.D.norm().item() / block.D.numel()
        return norms

    def compute_layer_norm(self, x, is_sequence=True, transpose=True):
        """ Compute the norms of the first item of the sequence, averaged over batches
        """

        with torch.no_grad():
            # average along the batch dimension
            y = x.mean(dim=0)
            # keep only the first item of the sequence
            if is_sequence:
                if transpose: y = y[..., 0]  # (H) --- if x is (B H L)
                else: y = y[0]

        return y.norm().item() / y.numel()

    def average_layer_norms(self, n_batches):
        for k in self.layer_norms.keys():
            self.layer_norms[k] /= n_batches

    def initialize_layer_norms(self):
        for k in self.layer_norms.keys():
            self.layer_norms[k] = 0.

    def compute_gradients(self, reduction='mean'):
        grads = {}
        for i, l in enumerate(self.dss_blocks):
            layer_grads = l.compute_gradients(reduction)
            grads.update(
                {'gradients/dss_layer_{}/{}'.format(i, k): v for k, v in layer_grads.items()}
            )
        return grads





class S4(DSS):

    def __init__(
        self,
        input_size,
        output_size,
        data_dim,
        state_size=DEFAULT_STATE_SIZE,
        kernel_length=None,
        bidirectional=False,
        activation='gelu',
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
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            data_dim=data_dim,
            state_size=state_size,
            bidirectional=bidirectional,
            activation=activation,
            kernel_version='exp',
            bias=bias,
            dropout=dropout,
            normalization=normalization,
            n_layers=n_layers,
            encoding=encoding,
            prenorm=prenorm,
            residual=residual,
            pooling=pooling,     # top pooling mode - 'last' or 'average' or 'manytomany'
            track_norms=track_norms,
            seed=seed,
            **kwargs
        )
        
        self.core_blocks = []

        for i in range(n_layers):
            # stacker n_layers blocks:
            # S4Layer (core) + activation + dropout + linear (mixing layer)
            core_block = nn.Sequential(OrderedDict([
                ('s4_layer', S4Layer(input_size=input_size, state_size=state_size, bidirectional=bidirectional, bias=bias, **kwargs)),
                ('activation', self.activation),
                ('dropout', self.drop),
                ('linear', nn.Linear(input_size, input_size, bias=bias))
            ]))
            setattr(self, f's4_block_{i}', core_block)
            self.core_blocks.append(core_block)
    
    def __str__(self):
        ret_str = str(self.input_layer) + "\n"
        ret_str += str(self.s4_block_0) + "\n"
        ret_str += "X {}".format(len(self.core_blocks)) + "\n"
        ret_str += str(self.post_norm) + "\n"
        ret_str += str(self.top_pooling) + "\n"
        ret_str += str(self.output_layer)
        return ret_str

    def compute_norms(self, L):
        """ Compute the norms of the first item of the kernels and of the matrics D
            of each S4 layer, for monitoring purposes
        """
        
        norms = {}
        with torch.no_grad():
            for i, block in enumerate(self.core_blocks):
                k = block.s4_layer.kernel(L)
                norms['norms/kernel_{}'.format(i)] = k[0].norm().item() / k[0].numel()
                norms['norms/D_{}'.format(i)] = block.s4_layer.D.norm().item() / block.s4_layer.D.numel()
        return norms
    




class TransformerEncoder(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        data_dim,
        seq_length,
        bias=True,
        encoding=None,
        dropout=0.1,
        pooling='last',     # top pooling mode - 'last' or 'average' or 'manytomany'
        seed=None,
        **kwargs
    ):
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        super().__init__()

        self.seq_length = seq_length
        self.input_size = input_size
        self.output_size = output_size
        if kwargs.get('padding_idx') is not None:
            self.padding_idx = kwargs.get('padding_idx')
        else:
            self.padding_idx = None

        #self.embedding = nn.Embedding(data_dim, input_size)
        self.input_layer = InputEncoder(data_dim, input_size, mode=encoding, **kwargs)
        self.pos_embedding = nn.Embedding(seq_length, input_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=8,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.output_layer = nn.Linear(input_size, output_size, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # top pooling layer
        self.top_pooling = TopPooling(mode=pooling)

    def generate_causal_mask(self, seq_length, device):
        return torch.triu(
            torch.ones(seq_length, seq_length, device=device),
            diagonal=1
        ).bool()

    def forward(self, u):
        #mask = self.generate_mask(u)
        mask = self.generate_causal_mask(self.seq_length, u.device)
        batch_size = u.shape[0]
        positions = torch.arange(0, self.seq_length, device=u.device).unsqueeze(0).expand(batch_size, self.seq_length)
        src_key_padding_mask = None if self.padding_idx is None else (u == self.padding_idx)

        x = self.dropout(
            self.input_layer(u) + self.pos_embedding(positions)
        )
        x = self.transformer(x, mask=mask, src_key_padding_mask=src_key_padding_mask)  # Pass the embeddings through the Transformer
        x = self.top_pooling(x)
        x = self.output_layer(x)
        return x




class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout
    ):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for l in self.encoder_layers:
            enc_output = l(enc_output, src_mask)

        dec_output = tgt_embedded
        for l in self.decoder_layers:
            dec_output = l(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output