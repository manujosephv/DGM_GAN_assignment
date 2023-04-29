import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class GeneratorMNISTSVHN(nn.Module):
    """Generator for transfering from mnist to svhn"""

    def __init__(self, conv_dim=64):
        super(GeneratorMNISTSVHN, self).__init__()
        # encoding blocks
        self.conv1 = conv(1, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.conv4 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)

        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)  # ( " )

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))  # (?, 3, 32, 32)
        return out


class GeneratorSVHNMNIST(nn.Module):
    """Generator for transfering from svhn to mnist"""

    def __init__(self, conv_dim=64):
        super(GeneratorSVHNMNIST, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.conv4 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)

        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)  # ( " )

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))  # (?, 1, 32, 32)
        return out


class DiscriminatorMNIST(nn.Module):
    """Discriminator for mnist."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(DiscriminatorMNIST, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out


class DiscriminatorSVHN(nn.Module):
    """Discriminator for svhn."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(DiscriminatorSVHN, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out


class CycleGAN(LightningModule):
    def __init__(
        self,
        g_conv_dim: int = 64,
        d_conv_dim: int = 64,
        n_discriminator_steps: int = 5,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        fixed_svhn: torch.Tensor = None,
        fixed_mnist: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.g_conv_dim = g_conv_dim
        self.d_conv_dim = d_conv_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.n_discriminator_steps = n_discriminator_steps
        if fixed_svhn.ndim == 2:
            fixed_svhn = fixed_svhn.unsqueeze(0)
        if fixed_mnist.ndim == 2:
            fixed_mnist = fixed_mnist.unsqueeze(0)
        self.fixed_svhn = fixed_svhn
        self.fixed_mnist = fixed_mnist

        """Builds a generator and a discriminator."""
        self.g_m_s = GeneratorMNISTSVHN(conv_dim=g_conv_dim)
        self.g_s_m = GeneratorSVHNMNIST(conv_dim=g_conv_dim)
        self.d_m = DiscriminatorMNIST(conv_dim=d_conv_dim, use_labels=False)
        self.d_s = DiscriminatorSVHN(conv_dim=d_conv_dim, use_labels=False)

    def forward(self, x, mnist_to_svhn=True):
        if mnist_to_svhn:
            return self.g_m_s(x)
        else:
            return self.g_s_m(x)

    def reset_grad(self, optimizer_g, optimizer_d):
        """Zeros the gradient buffers."""
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

    def merge_images(self, sources, targets, k=10):
        batch_size, _, h, w = sources.shape
        row = int(np.sqrt(batch_size))
        merged = np.zeros([3, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h : (i + 1) * h, (j * 2) * h : (j * 2 + 1) * h] = s
            merged[:, i * h : (i + 1) * h, (j * 2 + 1) * h : (j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)

    def training_step(self, batch, batch_idx):
        (mnist, _), (svhn, _) = batch

        optimizer_g, optimizer_d = self.optimizers()

        # Train discriminator
        ############################

        self.toggle_optimizer(optimizer_d)
        # Train with all-real batch
        optimizer_d.zero_grad()
        output = self.d_m(mnist)
        # Calculate loss on all-real batch
        d_m_loss = torch.mean((output - 1) ** 2)

        output = self.d_s(svhn)
        # Calculate loss on all-real batch
        d_s_loss = torch.mean((output - 1) ** 2)
        d_real_loss = d_m_loss + d_s_loss

        # Calculate gradients for D in backward pass
        self.manual_backward(d_real_loss)
        # Update D
        optimizer_d.step()
        # train with fake images
        self.reset_grad(optimizer_d, optimizer_g)
        # Generate a batch of images
        fake_svhn = self.g_m_s(mnist)
        output = self.d_s(fake_svhn)
        d_s_loss = torch.mean(output**2)

        fake_mnist = self.g_s_m(svhn)
        output = self.d_m(fake_mnist)
        d_m_loss = torch.mean(output**2)
        # Calculate D's loss on the all-fake batch
        d_fake_loss = d_m_loss + d_s_loss
        self.manual_backward(d_fake_loss)
        # Update D
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        # Training generator every n_discriminator steps
        if (batch_idx + 1) % self.n_discriminator_steps == 0:
            # train generator
            self.toggle_optimizer(optimizer_g)
            ############################
            # (2) Train the generator (Cycle training)
            ###########################
            # optimizer_g.zero_grad()
            self.reset_grad(optimizer_d, optimizer_g)
            # MNIST -> SVHN -> MNIST
            fake_svhn = self.g_m_s(mnist)
            output = self.d_s(fake_svhn)
            recon_mnist = self.g_s_m(fake_svhn)
            # g loss = d_s_loss + reconstruction loss
            g_loss = torch.mean((output - 1) ** 2) + torch.mean(
                (mnist - recon_mnist) ** 2
            )
            self.manual_backward(g_loss)
            optimizer_g.step()

            # SVHN -> MNIST -> SVHN
            self.reset_grad(optimizer_d, optimizer_g)
            fake_mnist = self.g_s_m(svhn)
            output = self.d_m(fake_mnist)
            recon_svhn = self.g_m_s(fake_mnist)
            # g loss = d_m_loss + reconstruction loss
            g_loss = torch.mean((output - 1) ** 2) + torch.mean(
                (svhn - recon_svhn) ** 2
            )
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

            self.log("d_real_loss", d_real_loss, prog_bar=False, logger=True)
            self.log("d_mnist_loss", d_m_loss, prog_bar=False, logger=True)
            self.log("d_svhn_loss", d_s_loss, prog_bar=True, logger=True)
            self.log("d_fake_loss", d_fake_loss, prog_bar=False, logger=True)
            self.log("g_loss", g_loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        g_params = list(self.g_m_s.parameters()) + list(self.g_s_m.parameters())
        d_params = list(self.d_m.parameters()) + list(self.d_s.parameters())

        opt_g = torch.optim.Adam(g_params, lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(d_params, lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_train_epoch_end(self):
        if self.fixed_svhn is not None and self.fixed_mnist is not None:
            fake_svhn = self.g_m_s(self.fixed_mnist.to(self.device))
            fake_mnist = self.g_s_m(self.fixed_svhn.to(self.device))
            mnist, fake_mnist = (
                self.fixed_mnist.cpu().numpy(),
                fake_mnist.detach().cpu().numpy(),
            )
            svhn, fake_svhn = (
                self.fixed_svhn.cpu().numpy(),
                fake_svhn.detach().cpu().numpy(),
            )
            merged = self.merge_images(mnist, fake_svhn)
            # split into list of images
            merged = [merged[i] for i in range(merged.shape[0])]
            self.logger.log_image("mnist_to_svhn", merged)
            merged = self.merge_images(svhn, fake_mnist)
            # split into list of images
            merged = [merged[i] for i in range(merged.shape[0])]
            self.logger.log_image("svhn_to_mnist", merged)
