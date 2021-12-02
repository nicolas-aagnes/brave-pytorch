import torch.nn as nn


class Projector(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_features: int = 4096
    ):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
        )

    def forward(self, x):
        return self.projector(x)


class Predictor(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_features: int = 4096
    ):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.predictor(x)