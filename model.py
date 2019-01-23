import torch
import torch.nn as nn
import torch.nn.functional as F


# A block consisting of convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
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

# A block consisting of a transposed convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
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

# A block consisting of an affine layer, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
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
    """
    Network Architecture:
        PRIOR OF Z:
            The prior of z is a Gaussian with mean and variance computed by the LSTM as follows
                h_t, c_t = prior_lstm(z_t-1, (h_t, c_t)) where h_t is the hidden state and c_t is the cell state
            Now the hidden state h_t is used to compute the mean and variance of z_t using an affine transform
                z_mean, z_log_variance = affine_mean(h_t), affine_logvar(h_t)
                z = reparameterize(z_mean, z_log_variance)
            The hidden state has dimension 512 and z has dimension 32

        CONVOLUTIONAL ENCODER:
            The convolutional encoder consists of 4 convolutional layers with 256 layers and a kernel size of 5 
            Each convolution is followed by a batch normalization layer and a LeakyReLU(0.2) nonlinearity. 
            For the 3,64,64 frames (all image dimensions are in channel, width, height) in the sprites dataset the following dimension changes take place
            
            3,64,64 -> 256,64,64 -> 256,32,32 -> 256,16,16 -> 256,8,8 (where each -> consists of a convolution, batch normalization followed by LeakyReLU(0.2))

            The 8,8,256 tensor is unrolled into a vector of size 8*8*256 which is then made to undergo the following tansformations
            
            8*8*256 -> 4096 -> 2048 (where each -> consists of an affine transformation, batch normalization followed by LeakyReLU(0.2))

        APPROXIMATE POSTERIOR FOR f:
            The approximate posterior is parameterized by a bidirectional LSTM that takes the entire sequence of transformed x_ts (after being fed into the convolutional encoder)
            as input in each timestep. The hidden layer dimension is 512

            Then the features from the unit corresponding to the last timestep of the forward LSTM and the unit corresponding to the first timestep of the 
            backward LSTM (as shown in the diagram in the paper) are concatenated and fed to two affine layers (without any added nonlinearity) to compute
            the mean and variance of the Gaussian posterior for f

        APPROXIMATE POSTERIOR FOR z (FACTORIZED q)
            Each x_t is first fed into an affine layer followed by a LeakyReLU(0.2) nonlinearity to generate an intermediate feature vector of dimension 512,
            which is then followed by two affine layers (without any added nonlinearity) to compute the mean and variance of the Gaussian Posterior of each z_t

            inter_t = intermediate_affine(x_t)
            z_mean_t, z_log_variance_t = affine_mean(inter_t), affine_logvar(inter_t)
            z = reparameterize(z_mean_t, z_log_variance_t)

        APPROXIMATE POSTERIOR FOR z (FULL q)
            The vector f is concatenated to each v_t where v_t is the encodings generated for each frame x_t by the convolutional encoder. This entire sequence  is fed into a bi-LSTM
            of hidden layer dimension 512. Then the features of the forward and backward LSTMs are fed into an RNN having a hidden layer dimension 512. The output h_t of each timestep
            of this RNN transformed by two affine transformations (without any added nonlinearity) to compute the mean and variance of the Gaussian Posterior of each z_t

            g_t = [v_t, f] for each timestep
            forward_features, backward_features = lstm(g_t for all timesteps)
            h_t = rnn([forward_features, backward_features])
            z_mean_t, z_log_variance_t = affine_mean(h_t), affine_logvar(h_t)
            z = reparameterize(z_mean_t, z_log_variance_t)

        CONVOLUTIONAL DECODER FOR CONDITIONAL DISTRIBUTION p(x_t | f, z_t)
            The architecture is symmetric to that of the convolutional encoder. The vector f is concatenated to each z_t, which then undergoes two subsequent
            affine transforms, causing the following change in dimensions
            
            256 + 32 -> 4096 -> 8*8*256 (where each -> consists of an affine transformation, batch normalization followed by LeakyReLU(0.2))

            The 8*8*256 tensor is reshaped into a tensor of shape 256,8,8 and then undergoes the following dimension changes 

            256,8,8 -> 256,16,16 -> 256,32,32 -> 256,64,64 -> 3,64,64 (where each -> consists of a transposed convolution, batch normalization followed by LeakyReLU(0.2)
            with the exception of the last layer that does not have batchnorm and uses tanh nonlinearity)

    Hyperparameters:
        f_dim: Dimension of the content encoding f. f has the shape (batch_size, f_dim)
        z_dim: Dimension of the dynamics encoding of a frame z_t. z has the shape (batch_size, frames, z_dim) 
        frames: Number of frames in the video. 
        hidden_dim: Dimension of the hidden states of the RNNs 
        nonlinearity: Nonlinearity used in convolutional and deconvolutional layers, defaults to LeakyReLU(0.2)
        in_size: Height and width of each frame in the video (assumed square)
        step: Number of channels in the convolutional and deconvolutional layers
        conv_dim: The convolutional encoder converts each frame into an intermediate encoding vector of size conv_dim, i.e,
                  The initial video tensor (batch_size, frames, num_channels, in_size, in_size) is converted to (batch_size, frames, conv_dim)
        factorised: Toggles between full and factorised posterior for z as discussed in the paper

    Optimization:
        The model is trained with the Adam optimizer with a learning rate of 0.0002, betas of 0.9 and 0.999, with a batch size of 25 for 200 epochs

    """
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
            self.z_inter = LinearUnit(self.conv_dim, self.hidden_dim, batchnorm=False)
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
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

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
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
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)

        return z_means, z_logvars, z_out


    def encode_frames(self, x):
        # The frames are unrolled into the batch dimension for batch processing such that x goes from
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x = x.view(-1, 3, self.in_size, self.in_size)
        x = self.conv(x)
        x = x.view(-1, self.step * (self.final_conv_size ** 2))
        x = self.conv_fc(x)
        # The frame dimension is reintroduced and x shape becomes [batch_size, frames, conv_dim]
        # This technique is repeated at several points in the code
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def decode_frames(self, zf):
        x = self.deconv_fc(zf)
        x = x.view(-1, self.step, self.final_conv_size, self.final_conv_size)
        x = self.deconv(x)
        return x.view(-1, self.frames, 3, self.in_size, self.in_size)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
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
            # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
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
