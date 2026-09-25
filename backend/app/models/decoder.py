import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):

    def __init__(
        self,
        feature_size,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers=1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Convert word IDs into embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Convert LSTM output to vocabulary scores
        self.linear = nn.Linear(
            hidden_size,
            vocab_size
        )

        # CNN features → initial hidden state
        self.init_h = nn.Linear(
            feature_size,
            hidden_size * num_layers
        )

        # CNN features → initial cell state
        self.init_c = nn.Linear(
            feature_size,
            hidden_size * num_layers
        )

    def forward(self, features, captions):

        # Convert token IDs into embeddings
        embeddings = self.embedding(captions)

        batch_size = features.size(0)

        # Initial hidden state
        h = self.init_h(features)

        # Initial cell state
        c = self.init_c(features)

        # Reshape:
        # [batch, hidden * layers]
        #
        # →
        #
        # [layers, batch, hidden]

        h = h.view(
            batch_size,
            self.num_layers,
            self.hidden_size
        )

        h = h.permute(1, 0, 2).contiguous()

        c = c.view(
            batch_size,
            self.num_layers,
            self.hidden_size
        )

        c = c.permute(1, 0, 2).contiguous()

        # Run LSTM
        outputs, _ = self.lstm(
            embeddings,
            (h, c)
        )

        # Convert to vocabulary scores
        outputs = self.linear(outputs)

        return outputs