import torch.nn as nn


class ConvGeneratorTranspose(nn.Module):
    def __init__(self, latent_sz, feature_map_sz, num_channels):
        super(ConvGeneratorTranspose, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_sz, feature_map_sz * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_sz * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                feature_map_sz * 8, feature_map_sz * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_map_sz * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                feature_map_sz * 4, feature_map_sz * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_map_sz * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature_map_sz * 2, feature_map_sz, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_sz),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(feature_map_sz * 2, feature_map_sz, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(feature_map_sz),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(feature_map_sz, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)
        for layer in self.main:
            z = layer(z)
        return z
        # return self.main(z)


class ConvGeneratorUpsample(nn.Module):
    def __init__(self, latent_sz, feature_map_sz, num_channels):
        super(ConvGeneratorUpsample, self).__init__()
        self.init_size = feature_map_sz // 4
        self.l1 = nn.Sequential(nn.Linear(latent_sz, 128 * self.init_size**2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, num_channels, 1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        for layer in self.conv_blocks:
            out = layer(out)
        return out
        # return self.conv_blocks(out)


class ConvDiscriminator(nn.Module):
    def __init__(self, num_channels, feature_map_sz, sigmoid: bool):
        super(ConvDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_channels, feature_map_sz, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(feature_map_sz, feature_map_sz * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_sz * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(feature_map_sz * 2, feature_map_sz * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_sz * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(feature_map_sz * 4, feature_map_sz * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_sz * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(feature_map_sz * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() if sigmoid else nn.Identity(),
        )

    def forward(self, input):
        for layer in self.main:
            input = layer(input)
        return input.view(input.size(0), -1)
        # return self.main(input).view(input.size(0), -1)
