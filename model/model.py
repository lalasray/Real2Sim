import torch
import torch.nn as nn
from model.vector_quant import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerprob
from model.enc_dec import IMUEncoder, IMUDecoder

class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._encoder = IMUEncoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(num_hiddens, embedding_dim, 1, 1)

        if decay > 0.0:
            self._vq_vae = VectorQuantizerprob(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self._decoder = IMUDecoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        z = z.unsqueeze(-1)
        loss, quantized, perplexity, code_probs = self._vq_vae(z)
        quantized = quantized.squeeze(-1)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity, code_probs