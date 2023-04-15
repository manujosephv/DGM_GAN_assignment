import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from pytorch_lightning import LightningModule

from models.modules import (
    ConvDiscriminator,
    ConvGeneratorTranspose,
    ConvGeneratorUpsample,
)

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGAN(LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        generator_feature_map_size: int = 64,
        discriminator_feature_map_size: int = 64,
        img_dimensions: tuple = (3, 32, 32),
        n_discriminator_steps: int = 5,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        upsample=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.latent_dim = latent_dim
        self.generator_feature_map_size = generator_feature_map_size
        self.discriminator_feature_map_size = discriminator_feature_map_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.n_discriminator_steps = n_discriminator_steps
        self._channels, self._height, self._width = img_dimensions

        # networks
        if upsample:
            self.generator = ConvGeneratorUpsample(
                latent_sz=self.latent_dim,
                feature_map_sz=generator_feature_map_size,
                num_channels=self._channels,
            )
        else:
            self.generator = ConvGeneratorTranspose(
                latent_sz=self.latent_dim,
                feature_map_sz=generator_feature_map_size,
                num_channels=self._channels,
            )
        self.generator.apply(weights_init)
        self.discriminator = ConvDiscriminator(
            num_channels=self._channels, feature_map_sz=discriminator_feature_map_size
        )
        self.discriminator.apply(weights_init)
        self.fixed_noise = torch.randn(64, latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs = batch

        # Adversarial Ground Truths
        valid = torch.ones(imgs.size(0), 1, dtype=imgs.dtype, device=imgs.device)
        fake = torch.zeros(imgs.size(0), 1, dtype=imgs.dtype, device=imgs.device)

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(
            imgs.shape[0], self.latent_dim, dtype=imgs.dtype, device=imgs.device
        )
        # generate images
        generated_imgs = self(z)

        # Training generator every n_discriminator steps
        if (batch_idx + 1) % self.n_discriminator_steps == 0:
            # train generator
            self.toggle_optimizer(optimizer_g)

            # ground truth result (ie: all fake)
            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.log("g_loss", g_loss, prog_bar=True)
            self.manual_backward(g_loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)

        # Train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake_loss = self.adversarial_loss(
            self.discriminator(generated_imgs.detach()), fake
        )

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_train_epoch_end(self):
        z = self.fixed_noise.to(self.device)
        # log sampled images
        sample_imgs = self(z)
        # grid = torchvision.utils.make_grid(sample_imgs)
        # split into list of images
        sample_imgs = [sample_imgs[i] for i in range(sample_imgs.shape[0])]
        self.logger.log_image("generated_images", sample_imgs)
