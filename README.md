# Disentangled Sequential Autoencoder
Reproduction of the ICML 2018 publication [Disentangled Sequential Autoencoder by Yinghen Li and Stephen Mandt](https://arxiv.org/abs/1803.02991), a Variational Autoencoder Architecture for learning latent representations of high dimensional sequential data by approximately disentangling the time invariant and the time variable features, without any modification to the ELBO objective. 

# Network Architecture

## Prior of z:

The prior of `z` is a Gaussian with mean and variance computed by the LSTM as follows
```
h_t, c_t = prior_lstm(z_t-1, (h_t, c_t)) where h_t is the hidden state and c_t is the cell state
```
Now the hidden state ```h_t``` is used to compute the mean and variance of ```z_t``` using an affine transform
```
z_mean, z_log_variance = affine_mean(h_t), affine_logvar(h_t)
z = reparameterize(z_mean, z_log_variance)
```
The hidden state has dimension 512 and z has dimension 32

## Convolutional Encoder:

The convolutional encoder consists of 4 convolutional layers with 256 layers and a kernel size of 5
Each convolution is followed by a batch normalization layer and a LeakyReLU(0.2) nonlinearity.
For the 3,64,64 frames (all image dimensions are in channel, width, height) in the sprites dataset the following dimension changes take place

```3,64,64 -> 256,64,64 -> 256,32,32 -> 256,16,16 -> 256,8,8 (where each -> consists of a convolution, batch normalization followed by LeakyReLU(0.2))```

The 8,8,256 tensor is unrolled into a vector of size ```8*8*256``` which is then made to undergo the following tansformations

```8*8*256 -> 4096 -> 2048 (where each -> consists of an affine transformation, batch normalization followed by LeakyReLU(0.2)) ```

## Approximate Posterior For f:

The approximate posterior is parameterized by a bidirectional LSTM that takes the entire sequence of transformed ```x_t```s (after being fed into the convolutional encoder)
as input in each timestep. The hidden layer dimension is 512

Then the features from the unit corresponding to the last timestep of the forward LSTM and the unit corresponding to the first timestep of the
backward LSTM (as shown in the diagram in the paper) are concatenated and fed to two affine layers (without any added nonlinearity) to compute
the mean and variance of the Gaussian posterior for f

## Approximate Posterior for z (Factorized q)

Each ```x_t``` is first fed into an affine layer followed by a LeakyReLU(0.2) nonlinearity to generate an intermediate feature vector of dimension 512,
which is then followed by two affine layers (without any added nonlinearity) to compute the mean and variance of the Gaussian Posterior of each ```z_t```

```
inter_t = intermediate_affine(x_t)
z_mean_t, z_log_variance_t = affine_mean(inter_t), affine_logvar(inter_t)
z = reparameterize(z_mean_t, z_log_variance_t)
```

## Approximate Posterior for z (FULL q)

The vector ```f``` is concatenated to each ```v_t``` where ```v_t``` is the encodings generated for each frame ```x_t``` by the convolutional encoder. This entire sequence  is fed into a bi-LSTM
of hidden layer dimension 512. Then the features of the forward and backward LSTMs are fed into an RNN having a hidden layer dimension 512. The output ```h_t``` of each timestep
of this RNN transformed by two affine transformations (without any added nonlinearity) to compute the mean and variance of the Gaussian Posterior of each ```z_t```

```
g_t = [v_t, f] for each timestep
forward_features, backward_features = lstm(g_t for all timesteps)
h_t = rnn([forward_features, backward_features])
z_mean_t, z_log_variance_t = affine_mean(h_t), affine_logvar(h_t)
z = reparameterize(z_mean_t, z_log_variance_t)
```

## Convolutional Decoder For Conditional Distribution 

The architecture is symmetric to that of the convolutional encoder. The vector ```f``` is concatenated to each ```z_t```, which then undergoes two subsequent
affine transforms, causing the following change in dimensions

```256 + 32 -> 4096 -> 8*8*256``` (where each -> consists of an affine transformation, batch normalization followed by LeakyReLU(0.2))

The ```8*8*256``` tensor is reshaped into a tensor of shape 256,8,8 and then undergoes the following dimension changes

```256,8,8 -> 256,16,16 -> 256,32,32 -> 256,64,64 -> 3,64,64``` (where each -> consists of a transposed convolution, batch normalization followed by LeakyReLU(0.2)
with the exception of the last layer that does not have batchnorm and uses tanh nonlinearity)

#  Optimizer
The model is trained with the Adam optimizer with a learning rate of 0.0002, betas of 0.9 and 0.999, with a batch size of 25 for 200 epochs

# Hyperparameters:

* Dimension of the content encoding f : 256 
* Dimension of the dynamics encoding of a frame z_t : 32 
* Number of frames in the video : 8
* Dimension of the hidden states of the RNNs : 512
* Nonlinearity used in convolutional and deconvolutional layers : LeakyReLU(0.2) in intermediate layers, Tanh in last layer of deconvolutional (Chosen arbitrarily, not stated in the paper)
* Number of channels in the convolutional and deconvolutional layers : 256
* Dimension of convolutional encoding generated from the video frames: 2048 (Chosen arbitrarily, not stated in the paper)
