import torch
import torch.nn as nn


class SpriteClassifier(nn.Module):
    def __init__(self,skin_colors, pants, tops, hairstyles, actions,
            num_frames=8, in_size=64, channels=64, code_dim=1024, hidden_dim=512, nonlinearity=None):
        super(SpriteClassifier, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        encoding_conv = []
        encoding_conv.append(nn.Sequential(nn.Conv2d(3, channels, 5, 4, 1, bias=False), nl))
        size = in_size // 4
        self.num_frames = num_frames
        while size > 4:
            encoding_conv.append(nn.Sequential(
                nn.Conv2d(channels, channels * 2, 5, 4, 1, bias=False),
                nn.BatchNorm2d(channels * 2), nl))
            size = size // 4
            channels *= 2
        self.encoding_conv = nn.Sequential(*encoding_conv)
        self.final_size = size
        self.final_channels = channels
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.encoding_fc = nn.Sequential(
                nn.Linear(size * size * channels, code_dim),
                nn.BatchNorm1d(code_dim), nl)
        # The last hidden state of a convolutional LSTM over the scenes is used for classification
        self.classifier_lstm = nn.LSTM(code_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.skin_color = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, skin_colors))
        self.pants = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, pants))
        self.top = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, tops))
        self.hairstyles = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, hairstyles))
        self.action = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, actions))

    def forward(self, x):
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.encoding_conv(x)
        x = x.view(-1, self.final_channels * (self.final_size ** 2))
        x = self.encoding_fc(x)
        x = x.view(-1, self.num_frames, self.code_dim)
        # Classifier output depends on last layer of LSTM: Can also change this to a bi-LSTM if required
        _, (hidden, _) = self.classifier_lstm(x)
        hidden = hidden.view(-1, self.hidden_dim)
        return self.skin_color(hidden), self.pants(hidden), self.top(hidden), self.hairstyles(hidden), self.action(hidden)
