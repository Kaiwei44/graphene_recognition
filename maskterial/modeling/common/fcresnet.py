import torch
from torch import nn

from .spec_norm import spectral_norm_fc


class FCResNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        embedding_dim: int = 16,
        depth: int = 4,
        spectral_normalization: bool = True,
        spec_coeff: float = 0.95,
        n_power_iterations: int = 1,
        dropout_rate: float = 0.01,
        num_classes: int = 2,
    ) -> torch.nn.Module:
        super().__init__()

        # ResFNN architecture from google paper: https://arxiv.org/abs/2006.10108
        self.first = nn.Linear(input_dim, embedding_dim)
        self.residuals = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for i in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)

        if spectral_normalization:
            self.first = spectral_norm_fc(
                self.first,
                coeff=spec_coeff,
                n_power_iterations=n_power_iterations,
            )

            for i in range(len(self.residuals)):
                self.residuals[i] = spectral_norm_fc(
                    self.residuals[i],
                    coeff=spec_coeff,
                    n_power_iterations=n_power_iterations,
                )
        self.last = nn.Linear(embedding_dim, num_classes)

        self.activation = nn.ELU()

    def get_embedding(self, x):
        x = self.first(x)
        for residual in self.residuals:
            x = x + self.activation(residual(x))
        return x

    def forward(self, x):
        x = self.first(x)
        for residual in self.residuals:
            x = x + self.dropout(self.activation(residual(x)))
        self.feature = x
        x = self.last(x)
        return x
