import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
"""
    The VAE code is based on
    https://github.com/pytorch/examples/blob/master/vae/main.py
    The q(x|z)-decoder is a bernoulli distribution rather than a Gaussian.
"""


class ELU_BatchNorm2d(torch.nn.Module):

    def __init__(self, filters):
        super(ELU_BatchNorm2d, self).__init__()
        self.bn = torch.nn.BatchNorm2d(filters)

    def forward(self, x):
        return self.bn(F.elu(x))


class ResidualBlock(torch.nn.Module):
    def __init__(self, filters, kernel_size):
        super(ResidualBlock, self).__init__()
        self.ops = torch.nn.Sequential(*[
            torch.nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            ELU_BatchNorm2d(filters),
            torch.nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            torch.nn.BatchNorm2d(filters)
        ])

    def forward(self, x):
        return F.elu(x + self.ops(x))


class Residual_AE(nn.Module):
    def __init__(self, dims, max_channels=1024, depth=7, n_hidden=1024):
        assert len(dims) == 3, 'Please specify 3 values for dims'
        super(Residual_AE, self).__init__()

        kernel_size = 3
        current_channels = 16
        self.epoch_factor = max(1, n_hidden//256)
        self.default_sigmoid = False
        self.netid = 'max.%d.d.%d.nH.%d'%(max_channels, depth, n_hidden)

        # encoder ###########################################
        modules = []
        spatial_sizes = [(dims[1], dims[2])]
        modules.append(torch.nn.Conv2d(in_channels=dims[0], out_channels=current_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False))
        modules.append(torch.nn.BatchNorm2d(current_channels))
        modules.append(torch.nn.ELU())
        for i in range(depth):
            modules.append(ResidualBlock(current_channels, kernel_size))
            next_channels = min(current_channels * 2, max_channels)
            modules.append(torch.nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=2, bias=False, padding=1))
            current_channels = next_channels
            modules.append(ELU_BatchNorm2d(current_channels))
            spatial_sizes.append(( math.floor(((spatial_sizes[-1][0]-1)/2) + 1), math.floor(((spatial_sizes[-1][1]-1)/2) + 1) ))

        # Bottleneck layer
        modules.append(ELU_BatchNorm2d(current_channels))
        modules.append(ResidualBlock(filters=current_channels, kernel_size=kernel_size))
        self.encoder = nn.Sequential(*modules)
        #
        # # decoder ###########################################
        modules = []
        out_pads = self._calculate_out_pad(spatial_sizes)
        for i in range(depth):
            next_channels = current_channels//2

            modules.append(torch.nn.ConvTranspose2d(current_channels, next_channels,
                                                    kernel_size=3, stride=2, bias=False, padding=1, output_padding=out_pads[i]))
            current_channels = next_channels
            modules.append(ELU_BatchNorm2d(current_channels))
            modules.append(ResidualBlock(current_channels, kernel_size))
        # Final layer
        modules.append(nn.Conv2d(current_channels, dims[0], kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False))
        self.decoder = nn.Sequential(*modules)

    def _calculate_out_pad(self, spatial_sizes, stride=2, padding=1, kernel_size=3, d=1):
        out_pad = []
        for i in reversed(range(1, len(spatial_sizes))):
            current = spatial_sizes[i]
            next = spatial_sizes[i-1]

            pad = [0, 0]
            for j in range(len(current)):
                outputsize = (current[j] - 1)*stride-2*padding + (kernel_size-1)*d + 1
                if outputsize < next[j]:
                    pad[j] = 1
            out_pad.append(pad)

        return out_pad

    def encode(self, x):
        n_samples = x.size(0)
        code = self.encoder(x)
        out = code.view(n_samples, -1)  # flatten to vectors.
        return out

    def forward(self, x, sigmoid=False):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        if sigmoid or self.default_sigmoid:
            dec = F.sigmoid(dec)
        return dec

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 240 * self.epoch_factor
        return config

    def preferred_name(self):
        return self.__class__.__name__+"."+self.netid


class Generic_AE(nn.Module):
    def __init__(self, dims, max_channels=512, depth=10, n_hidden=256):
        assert len(dims) == 3, 'Please specify 3 values for dims'
        super(Generic_AE, self).__init__()

        kernel_size = 3
        all_channels = []
        current_channels = 64
        nonLin = nn.ELU
        self.epoch_factor = max(1, n_hidden//256)
        self.default_sigmoid = False
        max_pool_layers = [i%2==0 for i in range(depth)]
        remainder_layers = []
        self.netid = 'max.%d.d.%d.nH.%d'%(max_channels, depth, n_hidden)

        # encoder ###########################################
        modules = []
        in_channels = dims[0]
        in_spatial_size = dims[1]
        for i in range(depth):
            modules.append(nn.Conv2d(in_channels, current_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2))
            modules.append(nn.BatchNorm2d(current_channels))
            modules.append(nonLin())
            in_channels = current_channels
            all_channels.append(current_channels)
            if max_pool_layers[i]:
                modules.append(nn.MaxPool2d(2))
                current_channels = min(current_channels * 2, max_channels)
                remainder_layers.append(in_spatial_size % 2)
                in_spatial_size = math.floor(in_spatial_size/2)

        # Final layer
        modules.append(nn.Conv2d(in_channels, n_hidden, kernel_size=kernel_size, padding=(kernel_size-1)//2))
        modules.append(nn.BatchNorm2d(n_hidden))
        modules.append(nonLin())
        self.encoder = nn.Sequential(*modules)

        # decoder ###########################################
        modules = []
        in_channels = n_hidden
        if self.__class__ == Generic_VAE:
            in_channels = in_channels // 2
        current_index = len(all_channels)-1
        r_ind = len(remainder_layers)-1
        for i in range(depth):
            modules.append(nn.Conv2d(in_channels, all_channels[current_index], kernel_size=kernel_size, padding=(kernel_size-1)//2))
            modules.append(nn.BatchNorm2d(all_channels[current_index]))
            modules.append(nonLin())
            if max_pool_layers[i]:
                modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
                if remainder_layers[r_ind] > 0:
                    modules.append(nn.ZeroPad2d((1,0,1,0)))
                r_ind -= 1 

            in_channels = all_channels[current_index]
            current_index -= 1
        # Final layer
        modules.append(nn.Conv2d(in_channels, dims[0], kernel_size=kernel_size, padding=(kernel_size-1)//2))
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        n_samples = x.size(0)
        code = self.encoder(x)
        out = code.view(n_samples, -1) # flatten to vectors.
        return out

    def forward(self, x, sigmoid=False):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        if sigmoid or self.default_sigmoid:
            dec = F.sigmoid(dec)
        return dec

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 240 * self.epoch_factor
        return config

    def preferred_name(self):
        return self.__class__.__name__+"."+self.netid

class Generic_VAE(Generic_AE):
    def __init__(self, dims, max_channels=512, depth=10, n_hidden=256):
        super(Generic_VAE, self).__init__(dims, max_channels, depth, 2*n_hidden)
        self.fc_e_mu  = nn.Linear(2*n_hidden, n_hidden)
        self.fc_e_std = nn.Linear(2*n_hidden, n_hidden)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
        n_samples = x.size(0)
        h_out = self.encoder(x)
        code  = self.fc_e_mu(h_out.view(n_samples, -1))
        return code

    def forward(self, x):
        enc     = self.encoder(x)
        n_size  = enc.size(0)
        mu, logvar  = self.fc_e_mu(enc.view(n_size, -1)), self.fc_e_std(enc.view(n_size, -1))
        self.last_mu  = mu
        self.last_std = logvar
        z           = self.reparameterize(mu, logvar)        
        dec = self.decoder(z.view(n_size, enc.size(1)//2, enc.size(2), enc.size(3)))
        dec = F.sigmoid(dec)
        return dec


class ALILikeAE(nn.Module):
    def __init__(self, dims, max_channels=1024, depth=6, n_hidden=512):
        assert len(dims) == 3, 'Please specify 3 values for dims'
        super(ALILikeAE, self).__init__()

        EncKernel = [2, 7, 5, 7, 4, 1]
        EncStride = [1, 2, 2, 2, 1, 1]
        EncDepth = [64, 128, 256, 512, 512, n_hidden]

        # Generator param
        GenKernel = [4, 7, 5, 7, 2, 1]
        GenStride = [1, 2, 2, 2, 1, 1]
        GenDepth = [256, 128, 64, 32, 32, dims[0]]

        self.epoch_factor = max(1, n_hidden//256)
        self.default_sigmoid = False

        remainder_layers = []
        self.netid = 'max.%d.d.%d.nH.%d'%(max_channels, depth, n_hidden)

        # encoder ###########################################
        modules = []
        in_channels = dims[0]

        for i in range(depth):
            modules.append(nn.Conv2d(in_channels, EncDepth[i], kernel_size=EncKernel[i], padding=0, stride=EncStride[i]))
            modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
            modules.append(nn.BatchNorm2d(EncDepth[i]))
            in_channels = EncDepth[i]

        self.encoder = nn.Sequential(*modules)
        self.fc_e_mu = nn.Linear(2 * n_hidden, n_hidden)
        self.fc_e_std = nn.Linear(2 * n_hidden, n_hidden)
        # decoder ###########################################
        modules = []
        in_channels = n_hidden
        if self.__class__ == Generic_VAE:
            in_channels = in_channels // 2

        for i in range(depth):
            modules.append(nn.ConvTranspose2d(in_channels, GenDepth[i], kernel_size=GenKernel[i], padding=0, stride=GenStride[i]))
            modules.append(torch.nn.ReLU(True))
            modules.append(nn.BatchNorm2d(GenDepth[i]))
            in_channels = GenDepth[i]
        # Final layer
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        n_samples = x.size(0)
        code = self.encoder(x)
        out = code.view(n_samples, -1) # flatten to vectors.
        return out

    def forward(self, x, sigmoid=False):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        if sigmoid or self.default_sigmoid:
            dec = (F.tanh(dec) + 1) / 2.        # It's actually TanhHHHHHH
        return dec

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 240 * self.epoch_factor
        return config

    def preferred_name(self):
        return self.__class__.__name__+"."+self.netid


class ALILikeVAE(nn.Module):
    def __init__(self, dims, max_channels=1024, depth=6, n_hidden=512):
        assert len(dims) == 3, 'Please specify 3 values for dims'
        super(ALILikeVAE, self).__init__()

        EncKernel = [2, 7, 5, 7, 4]
        EncStride = [1, 2, 2, 2, 1]
        EncDepth = [64, 128, 256, 512, 512]

        # Generator param
        GenKernel = [4, 7, 5, 7, 2, 1]
        GenStride = [1, 2, 2, 2, 1, 1]
        GenDepth = [256, 128, 64, 32, 32, dims[0]]

        self.epoch_factor = max(1, n_hidden//256)
        self.default_sigmoid = False

        remainder_layers = []
        self.netid = 'max.%d.d.%d.nH.%d'%(max_channels, depth, n_hidden)

        # encoder ###########################################
        modules = []
        in_channels = dims[0]

        for i in range(depth-1):
            modules.append(nn.Conv2d(in_channels, EncDepth[i], kernel_size=EncKernel[i], padding=0, stride=EncStride[i]))
            modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
            modules.append(nn.BatchNorm2d(EncDepth[i]))
            in_channels = EncDepth[i]

        self.fc_e_mu = nn.Linear(in_channels, n_hidden)
        self.fc_e_std = nn.Linear(in_channels, n_hidden)
        self.encoder = nn.Sequential(*modules)

        # decoder ###########################################
        modules = []
        in_channels = n_hidden
        if self.__class__ == Generic_VAE:
            in_channels = in_channels // 2

        for i in range(depth):
            modules.append(nn.ConvTranspose2d(in_channels, GenDepth[i], kernel_size=GenKernel[i], padding=0, stride=GenStride[i]))
            modules.append(torch.nn.ReLU(True))
            modules.append(nn.BatchNorm2d(GenDepth[i]))
            in_channels = GenDepth[i]
        # Final layer
        self.decoder = nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
        n_samples = x.size(0)
        h_out = self.encoder(x)
        code  = self.fc_e_mu(h_out.view(n_samples, -1))
        return code

    def forward(self, x):
        enc     = self.encoder(x)
        n_size  = enc.size(0)
        mu, logvar  = self.fc_e_mu(enc.view(n_size, -1)), self.fc_e_std(enc.view(n_size, -1))
        self.last_mu  = mu
        self.last_std = logvar
        z           = self.reparameterize(mu, logvar)
        dec = self.decoder(z.view(n_size, -1, enc.size(2), enc.size(3)))
        dec = (F.tanh(dec) + 1.0)/ 2.0
        return dec

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 240 * self.epoch_factor
        return config

    def preferred_name(self):
        return self.__class__.__name__+"."+self.netid

class ALILikeResAE(nn.Module):
    def __init__(self, dims, max_channels=1024, depth=6, n_hidden=512):
        assert len(dims) == 3, 'Please specify 3 values for dims'
        super(ALILikeResAE, self).__init__()

        EncKernel = [2, 7, 5, 7, 4, 1]
        EncStride = [1, 2, 2, 2, 1, 1]
        EncDepth = [64, 128, 256, 512, 512, n_hidden]

        # Generator param
        GenKernel = [4, 7, 5, 7, 2, 1]
        GenStride = [1, 2, 2, 2, 1, 1]
        GenDepth = [256, 128, 64, 32, 32, dims[0]]

        self.epoch_factor = max(1, n_hidden//256)
        self.default_sigmoid = False

        remainder_layers = []
        self.netid = 'max.%d.d.%d.nH.%d'%(max_channels, depth, n_hidden)

        # encoder ###########################################
        modules = []
        in_channels = dims[0]

        for i in range(depth):
            modules.append(nn.Conv2d(in_channels, EncDepth[i], kernel_size=EncKernel[i], padding=0, stride=EncStride[i]))
            modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
            modules.append(nn.BatchNorm2d(EncDepth[i]))
            modules.append(ResidualBlock(EncDepth[i], 3))
            modules.append(ELU_BatchNorm2d(EncDepth[i]))
            in_channels = EncDepth[i]

        self.encoder = nn.Sequential(*modules)
        self.fc_e_mu = nn.Linear(2 * n_hidden, n_hidden)
        self.fc_e_std = nn.Linear(2 * n_hidden, n_hidden)
        # decoder ###########################################
        modules = []
        in_channels = n_hidden
        if self.__class__ == Generic_VAE:
            in_channels = in_channels // 2

        for i in range(depth):
            modules.append(nn.ConvTranspose2d(in_channels, GenDepth[i], kernel_size=GenKernel[i], padding=0, stride=GenStride[i]))
            modules.append(torch.nn.ReLU(True))
            modules.append(nn.BatchNorm2d(GenDepth[i]))
            modules.append(ResidualBlock(GenDepth[i], 3))
            modules.append(ELU_BatchNorm2d(GenDepth[i]))
            in_channels = GenDepth[i]
        # Final layer
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        n_samples = x.size(0)
        code = self.encoder(x)
        out = code.view(n_samples, -1) # flatten to vectors.
        return out

    def forward(self, x, sigmoid=False):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        if sigmoid or self.default_sigmoid:
            dec = (F.tanh(dec) + 1) / 2.        # It's actually TanhHHHHHH
        return dec

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 240 * self.epoch_factor
        return config

    def preferred_name(self):
        return self.__class__.__name__+"."+self.netid

class ALILikeResVAE(nn.Module):
    def __init__(self, dims, max_channels=1024, depth=6, n_hidden=512):
        assert len(dims) == 3, 'Please specify 3 values for dims'
        super(ALILikeResVAE, self).__init__()

        EncKernel = [2, 7, 5, 7, 4]
        EncStride = [1, 2, 2, 2, 1]
        EncDepth = [64, 128, 256, 512, 512]

        # Generator param
        GenKernel = [4, 7, 5, 7, 2, 1]
        GenStride = [1, 2, 2, 2, 1, 1]
        GenDepth = [256, 128, 64, 32, 32, dims[0]]

        self.epoch_factor = max(1, n_hidden//256)
        self.default_sigmoid = False

        remainder_layers = []
        self.netid = 'max.%d.d.%d.nH.%d'%(max_channels, depth, n_hidden)

        # encoder ###########################################
        modules = []
        in_channels = dims[0]

        for i in range(depth-1):
            modules.append(nn.Conv2d(in_channels, EncDepth[i], kernel_size=EncKernel[i], padding=0, stride=EncStride[i]))
            modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
            modules.append(nn.BatchNorm2d(EncDepth[i]))
            modules.append(ResidualBlock(EncDepth[i], 3))
            modules.append(ELU_BatchNorm2d(EncDepth[i]))
            in_channels = EncDepth[i]

        self.fc_e_mu = nn.Linear(in_channels, n_hidden)
        self.fc_e_std = nn.Linear(in_channels, n_hidden)
        self.encoder = nn.Sequential(*modules)

        # decoder ###########################################
        modules = []
        in_channels = n_hidden
        if self.__class__ == Generic_VAE:
            in_channels = in_channels // 2

        for i in range(depth):
            modules.append(nn.ConvTranspose2d(in_channels, GenDepth[i], kernel_size=GenKernel[i], padding=0, stride=GenStride[i]))
            modules.append(torch.nn.ReLU(True))
            modules.append(nn.BatchNorm2d(GenDepth[i]))
            modules.append(ResidualBlock(GenDepth[i], 3))
            modules.append(ELU_BatchNorm2d(GenDepth[i]))
            in_channels = GenDepth[i]
        # Final layer
        self.decoder = nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
        n_samples = x.size(0)
        h_out = self.encoder(x)
        code  = self.fc_e_mu(h_out.view(n_samples, -1))
        return code

    def forward(self, x):
        enc     = self.encoder(x)
        n_size  = enc.size(0)
        mu, logvar  = self.fc_e_mu(enc.view(n_size, -1)), self.fc_e_std(enc.view(n_size, -1))
        self.last_mu  = mu
        self.last_std = logvar
        z           = self.reparameterize(mu, logvar)
        dec = self.decoder(z.view(n_size, -1, enc.size(2), enc.size(3)))
        dec = (F.tanh(dec) + 1.0)/ 2.0
        return dec

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 240 * self.epoch_factor
        return config

    def preferred_name(self):
        return self.__class__.__name__+"."+self.netid

class VAE_Loss(nn.Module):
    def __init__(self, VAE_model, BCE):
        super(VAE_Loss, self).__init__()
        #assert VAE_model.__class__ == Generic_VAE, 'Only Generic_VAEs are accepted.'
        self.VAE = VAE_model
        self.size_average = True
        self.reduction = 'sum'
        if BCE:
            self.loss = nn.BCELoss(size_average=False)
        else:
            self.loss = nn.MSELoss(size_average=False)

    def forward(self, X, Y):
        BCE_loss = self.loss(X, Y)
        mu, logvar = self.VAE.last_mu, self.VAE.last_std
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE_loss + KLD)/X.numel()
