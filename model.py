import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DisentangledVAE']


# Differences from original paper : Step up the number of filters in multiples of 64
class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
                    nn.BatchNorm2d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding), nonlinearity)

    def forward(self, x):
        return self.model(x)

class ConvUnitTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, out_padding=0, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnitTranspose, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding),
                    nn.BatchNorm2d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding), nonlinearity)

    def forward(self, x):
        return self.model(x)

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)


class DisentangledVAE(nn.Module):
    def __init__(self, f_dim=256, z_dim=32, conv_dim=2048, step=256, in_size=64, hidden_dim=512,
                 frames=8, nonlinearity=None, factorised=False, device=torch.device('cpu')):
        super(DisentangledVAE, self).__init__()
        self.device = device
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.factorised = factorised
        self.step = step
        self.in_size = in_size
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        # TODO: Check if only one affine transform is sufficient. Paper says distribution is parameterised by LSTM
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        if self.factorised is True:
            # Paper says : 1 Hidden Layer MLP. Last layers shouldn't have any nonlinearities
            self.z_inter = LinearUnit(self.conv_dim, self.conv_dim // 4, batchnorm=False)
            self.z_mean = nn.Linear(self.conv_dim // 4, self.z_dim)
            self.z_logvar = nn.Linear(self.conv_dim // 4, self.z_dim)
        else:
            # TODO: Check if one affine transform is sufficient. Paper says distribution is parameterised by RNN over LSTM. Last layer shouldn't have any nonlinearities
            self.z_lstm = nn.LSTM(self.conv_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
            self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
            # Each timestep is for each z so no reshaping and feature mixing
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.conv = nn.Sequential(
                ConvUnit(3, step, 5, 1, 2), # 3*64*64 -> 256*64*64
                ConvUnit(step, step, 5, 2, 2), # 256,64,64 -> 256,32,32
                ConvUnit(step, step, 5, 2, 2), # 256,32,32 -> 256,16,16
                ConvUnit(step, step, 5, 2, 2), # 256,16,16 -> 256,8,8
                )
        self.final_conv_size = in_size // 8
        self.conv_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))

        self.deconv_fc = nn.Sequential(LinearUnit(self.f_dim + self.z_dim, self.conv_dim * 2, False),
                LinearUnit(self.conv_dim * 2, step * (self.final_conv_size ** 2), False))
        self.deconv = nn.Sequential(
                ConvUnitTranspose(step, step, 5, 2, 2, 1),
                ConvUnitTranspose(step, step, 5, 2, 2, 1),
                ConvUnitTranspose(step, step, 5, 2, 2, 1),
                ConvUnitTranspose(step, 3, 5, 1, 2, 0, nonlinearity=nn.Tanh()))

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def sample_z(self, batch_size, random_sampling=True):
        z_out = None
        z_means = None
        z_logvars = None

        z_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        z_mean_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        z_logvar_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for _ in range(self.frames):
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)

        return z_means, z_logvars, z_out


    def encode_frames(self, x):
        x = x.view(-1, 3, self.in_size, self.in_size)
        x = self.conv(x)
        x = x.view(-1, self.step * (self.final_conv_size ** 2))
        x = self.conv_fc(x)
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def decode_frames(self, zf):
        x = self.deconv_fc(zf)
        x = x.view(-1, self.step, self.final_conv_size, self.final_conv_size)
        x = self.deconv(x)
        return x.view(-1, self.frames, 3, self.in_size, self.in_size)

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode_z(self, x, f):
        if self.factorised is True:
            features = self.z_inter(x)
        else:
            f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
            lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim=2))
            features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def forward(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encode_frames(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)
        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decode_frames(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x
