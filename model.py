import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DisentangledVAE']


# Differences from original paper : Step up the number of filters in multiples of 64
class DisentangledVAE(nn.Module):
    def __init__(self, f_dim=64, z_dim=32, conv_dim=1024, step=64, in_size=64, hidden_dim=512,
                 frames=8, nonlinearity=None, factorised=False):
        super(DisentangledVAE, self).__init__()
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.factorised = factorised
        self.step = step
        self.in_size = in_size
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        self.f_mean = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.f_dim), nl)
        self.f_logvar = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.f_dim), nl)

        if self.factorised is True:
            self.z_mean = nn.Sequential(nn.Linear(self.conv_dim, self.z_dim), nl)
            self.z_logvar = nn.Sequential(nn.Linear(self.conv_dim, self.z_dim), nl)
        else:
            self.z_lstm = nn.LSTM(self.conv_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
            self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
            self.z_mean = nn.Sequential(nn.Linear(self.hidden_dim, self.z_dim), nl)
            self.z_logvar = nn.Sequential(nn.Linear(self.hidden_dim, self.z_dim), nl)

        encoding_conv = []
        channels = step
        encoding_conv.append(nn.Sequential(nn.Conv2d(3, channels, 5, 4, 1, bias=False), nl))
        size = in_size // 4
        while size > 4:
            encoding_conv.append(nn.Sequential(
                nn.Conv2d(channels, channels * 2, 5, 4, 1, bias=False),
                nn.BatchNorm2d(channels * 2), nl))
            size = size // 4
            channels *= 2

        self.encode_final_size = size
        self.encode_final_channels = channels

        self.conv = nn.Sequential(*encoding_conv)  # Common to q(f|x1:T) and q(z1:T)
        self.conv_fc = nn.Sequential(
                nn.Linear(size * size * channels, conv_dim),
                nn.BatchNorm1d(conv_dim), nl)

        self.deconv_fc = nn.Sequential(nn.Linear(self.f_dim + self.z_dim, 4 * 4 * step), nl)
        decoding_conv = []
        channels = step
        size = 4
        while size < in_size // 4:
            decoding_conv.append(nn.Sequential(
                nn.ConvTranspose2d(channels, channels * 2, 5, 4, 1, 1),
                nn.BatchNorm2d(channels * 2), nl))
            channels *= 2
            size *= 4
        decoding_conv.append(nn.Sequential(
            nn.ConvTranspose2d(channels, 3, 5, 4, 1, 1), nn.Tanh()))
        self.deconv = nn.Sequential(*decoding_conv)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
#        nn.init.xavier_normal_(self.deconv[len(self.deconv) - 2].weight, nn.init.calculate_gain('tanh'))

    def encode_frames(self, x):
        x = x.view(-1, 3, self.in_size, self.in_size)
        x = self.conv(x)
        x = x.view(-1, self.encode_final_channels * (self.encode_final_size ** 2))
        x = self.conv_fc(x)
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def decode_frames(self, zf):
        x = zf.view(-1, self.f_dim+self.z_dim)
        x = self.deconv_fc(x)
        x = x.view(-1, self.step, 4, 4)
        x = self.deconv(x)
        return x.view(-1, self.frames, 3, self.in_size, self.in_size)

    def reparameterize(self, mean, logvar):
        if self.training:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        mean = self.f_mean(lstm_out[:, self.frames-1])
        logvar = self.f_logvar(lstm_out[:, self.frames-1])
        return mean, logvar, self.reparameterize(mean, logvar)

    def encode_z(self, x, f):
        if self.factorised is True:
            features = x
        else:
            f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
            lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim=2))
            features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar)

    def forward(self, x):
        conv_x = self.encode_frames(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)
        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decode_frames(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, recon_x
