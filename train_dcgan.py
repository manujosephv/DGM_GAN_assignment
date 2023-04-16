import wandb
import uuid
import pytorch_lightning as pl

from models.dcgan import DCGAN
from datamodules import BitMojiDataModule


def train():
    wandb.login(key="1a6be1c224ac6e7582f1f314f2409a5d403324a0")
    UPSAMPLE = False
    dm = BitMojiDataModule(
        root_dir="data/bitmojis",
        batch_size=64,
        num_workers=4,
        size=64,
    )
    dm.setup()
    model = DCGAN(
        latent_dim=100,
        generator_feature_map_size=64,
        discriminator_feature_map_size=64,
        img_dimensions=(3, 64, 64),
        n_discriminator_steps=1,
        lr=1e-4,
        b1=0.5,  # Momentum based does not do well. Plan to change
        b2=0.999,
        upsample=UPSAMPLE,
    )
    tag = "upsampled" if UPSAMPLE else "transpose"
    uid = uuid.uuid4()
    # Wandb logger
    wandb_logger = pl.loggers.WandbLogger(project="GAN", name=f"DCGAN_{tag}_{uid}")
    # es = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="g_loss",
        dirpath="checkpoints",
        filename=f"DCGAN_{tag}_{uid}" + "-{epoch:02d}-{val_loss:.2f}",
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
    # pl.seed_everything(SEED)
    train()
