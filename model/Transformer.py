import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = "Transformer"

        self.seq_len = self.args.transformer_input_length
        dropout = self.args.transformer_dropout
        nhead = self.args.transformer_nhead
        nlayers = self.args.transformer_nlayers
        output_dim = self.args.output_dim

        self.padding_value = 0
        d_model = self.args.whisper_dim

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.mlp = nn.Linear(d_model, output_dim)

    def encode(self, whisper):
        """
        Args:
            whisper: Tensor, shape [real_whisper_len]
        Returns:
            input: Tensor, shape [seq_len]
            mask: Tensor, shape [seq_len]
        """
        input = torch.zeros(
                (self.seq_len, self.args.whisper_dim),
                device=self.args.device,
                dtype=torch.float,
            )

        mask = torch.ones(self.seq_len, device=self.args.device, dtype=torch.long)

        whisper = whisper[: self.seq_len]
        input[: len(whisper)] = torch.as_tensor(whisper, device=self.args.device)
        padding_len = self.seq_len - len(whisper)

        if padding_len > 0:
            input[-padding_len:] = self.padding_value
            mask[-padding_len:] = 0

        return input, mask

    def forward(self, idxs, dataset):
        """
        Args:
            idxs: Tensor, shape [batch_size]
            dataset: DatasetLoader
        Returns:
            output: Tensor, shape [batch_size, seq_len, output_dim]
            masks: Tensor, shape [batch_size, seq_len, 1]
        """
        inputs = [self.encode(dataset.whisper[idx.item()]) for idx in idxs]

        # (bs, seq_len, whisper_dim)
        input = torch.stack([i[0] for i in inputs])
        # (bs, seq_len, 1)
        masks = torch.stack([i[1] for i in inputs])[:, :, None]

        # (bs, seq_len, whisper_dim), where whisper_dim = d_model
        src = self.pos_encoder(input)
        output = self.transformer_encoder(src)
        # (bs, seq_len, output_dim)
        output = self.mlp(output)
        return output, masks


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
