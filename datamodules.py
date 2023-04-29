from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import SVHN, MNIST
import pytorch_lightning as pl
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
pl.seed_everything(42)
rng = np.random.default_rng(42)


# pytorch data set to load bitmoji dataset
class BitMojiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)
        self.files = [os.path.join(root_dir, f) for f in self.files]
        self.files = [f for f in self.files if f.endswith(".png")]
        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image


# pytorch lightning data module to load bitmoji dataset
class BitMojiDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=64, num_workers=0, size=128):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((self.size, self.size)),
                    # scale between -1 and 1
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
            self.dataset = BitMojiDataset(self.root_dir, transform)
            # Create data indices for training and validation splits:
            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(0.2 * dataset_size))
            train_indices, val_indices = indices[:split], indices[split:]
            rng.shuffle(val_indices)
            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# PyTorch dataset to load image pairs for cycle gan
class ImagePairDataset(Dataset):
    def __init__(self, dataset_a, dataset_b):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

    def __len__(self):
        return min(len(self.dataset_a), len(self.dataset_b))

    def __getitem__(self, idx):
        image_a, lbl_1 = self.dataset_a[idx]
        image_b, lbl_2 = self.dataset_b[idx]
        return (image_a, lbl_1), (image_b, lbl_2)


# pytorch lightning data module to load SVHN-MNIST dataset pair
class MNISTSVHNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=0,
        size=128,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.root_dir = "data/"

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # create transforms
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((self.size, self.size)),
                    # scale between -1 and 1
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
            self.svhn_dataset_train = SVHN(
                self.root_dir, split="train", download=True, transform=transform
            )
            self.svhn_dataset_val = SVHN(
                self.root_dir, split="test", download=True, transform=transform
            )
            self.mnist_dataset_train = MNIST(
                self.root_dir, train=True, download=True, transform=transform
            )
            self.mnist_dataset_val = MNIST(
                self.root_dir, train=False, download=True, transform=transform
            )
            self.train_dataset = ImagePairDataset(
                self.mnist_dataset_train, self.svhn_dataset_train
            )
            self.val_dataset = ImagePairDataset(
                self.mnist_dataset_val, self.svhn_dataset_val
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    # test bitmoji data module
    # dm = BitMojiDataModule(
    #     root_dir="data/bitmojis",
    #     batch_size=64,
    #     num_workers=0,
    #     size=128,
    # )
    # dm.setup()
    # print(len(dm.train_dataset))
    # print(len(dm.val_dataset))
    # # print first batch
    # for batch in dm.train_dataloader():
    #     print(batch[0].shape)
    #     print(batch[1].shape)
    #     break
    # test svhn-mnist data module
    dm = MNISTSVHNDataModule(
        batch_size=64,
        num_workers=0,
        size=32,
    )
    dm.setup()
    print(len(dm.train_dataset))
    print(len(dm.val_dataset))
    # print first batch
    for batch in dm.train_dataloader():
        (img1, lbl1), (img2, lbl2) = batch
        print(img1[0].shape)
        print(img2[1].shape)
        break
