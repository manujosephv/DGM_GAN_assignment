import wandb
import uuid
import pytorch_lightning as pl

from models.cyclegan import CycleGAN
from datamodules import MNISTSVHNDataModule


def train():
    wandb.login(key="1a6be1c224ac6e7582f1f314f2409a5d403324a0")
    dm = MNISTSVHNDataModule(
        batch_size=64,
        num_workers=4,
        size=32,
    )
    dm.setup()
    train_dl = dm.train_dataloader()
    # get first batch
    for (fixed_mnist, _), (fixed_svhn, _) in train_dl:
        break
    model = CycleGAN(
        g_conv_dim=64,
        d_conv_dim=64,
        n_discriminator_steps=1,
        lr=0.0002,
        b1=0.5,  # Momentum based does not do well. Plan to change
        b2=0.999,
        fixed_mnist=fixed_mnist,
        fixed_svhn=fixed_svhn,
    )
    uid = uuid.uuid4()
    # Wandb logger to track experiment in Weights and Biases
    wandb_logger = pl.loggers.WandbLogger(project="GAN", name=f"CycleGAN_{uid}")
    # es = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    # To save best model and will be saved with same uid as the WANDB experiment
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="g_loss",
        dirpath="checkpoints",
        filename=f"CycleGAN_{uid}" + "-{epoch:02d}-{g_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        accelerator="gpu",  # "gpu",
        callbacks=[checkpoint_callback],
        max_epochs=10,
        logger=wandb_logger,
        fast_dev_run=False,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    pl.seed_everything(42)
    train()
