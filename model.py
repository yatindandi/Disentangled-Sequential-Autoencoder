import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DisentangledVAE']


class DisentangledVAE(nn.Module):
    def __init__(self, f_dim=64, z_dim=32, conv_dim=1024, hidden_dim=512,
                 frames=8, filters=256, p_drop=0.5, nonlinearity=None, factorised=False):
        super(DisentangledVAE, self).__init__()
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.factorised = factorised
        self.filters=filters
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        self.f_mean = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(self.hidden_dim * 2, self.f_dim))
        self.f_logvar = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(self.hidden_dim * 2, self.f_dim)
                )

        if self.factorised is True:
            self.z_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                                  bidirectional=True, batch_first=True)
        else:
            self.z_lstm = nn.LSTM(self.conv_dim+self.f_dim, self.hidden_dim, 1,
                                  bidirectional=True, batch_first=True)

        self.z_rnn = nn.RNN(self.hidden_dim*2, self.hidden_dim, batch_first=True)

        self.z_mean = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(self.hidden_dim, self.z_dim)
                )
        self.z_logvar = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(self.hidden_dim, self.z_dim)
                )

        self.conv = nn.Sequential(
                nn.Conv2d(3, filters, 4, 2, 1, bias=True), nl,
                nn.Conv2d(filters, filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(filters), nl,
                nn.Dropout2d(p_drop),
                nn.Conv2d(filters, filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(filters), nl,
                nn.Dropout2d(p_drop),
                nn.Conv2d(filters, filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(filters), nl,
                nn.Dropout2d(p_drop))

        self.conv_fc = nn.Sequential(
                nn.Linear(4*4*filters, self.conv_dim),
                nn.BatchNorm1d(self.conv_dim), nl)

        self.deconv_fc = nn.Sequential(
                nn.Linear(self.f_dim+self.z_dim, 4*4*filters),
                nn.BatchNorm1d(4*4*filters), nl)

        self.deconv = nn.Sequential(
                nn.ConvTranspose2d(filters, filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(filters), nl,
                nn.Dropout2d(p_drop),
                nn.ConvTranspose2d(filters, filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(filters), nl,
                nn.Dropout2d(p_drop),
                nn.ConvTranspose2d(filters, filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(filters), nl,
                nn.Dropout2d(p_drop),
                nn.ConvTranspose2d(filters, 3, 4, 2, 1, bias=True), nn.Tanh())

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        nn.init.xavier_normal_(self.deconv[len(self.deconv) - 2].weight, nn.init.calculate_gain('tanh'))

    def encode_frames(self, x):
        x = x.view(-1, 3, 64, 64)
        x = self.conv(x)
        x = x.view(-1, 4*4*self.filters)
        x = self.conv_fc(x)
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def decode_frames(self, zf):
        x = zf.view(-1, self.f_dim+self.z_dim)
        x = self.deconv_fc(x)
        x = x.view(-1, self.filters, 4, 4)
        x = self.deconv(x)
        print(x.shape)
        return x.view(-1, self.frames, 3, 64, 64)

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
            lstm_out, _ = self.z_lstm(x)
        else:
            f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
            lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim=2))
        rnn_out, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(rnn_out)
        logvar = self.z_logvar(rnn_out)
        return mean, logvar, self.reparameterize(mean, logvar)

    def forward(self, x):
        conv_x = self.encode_frames(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)
        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decode_frames(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, recon_x
