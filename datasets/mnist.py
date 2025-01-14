import lightning as L
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", train_batch_size: int = 32, val_batch_size: int = 32, train_workers: int = 4, val_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [45000, 15000]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.train_batch_size, num_workers=self.train_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.val_batch_size, num_workers=self.val_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.val_batch_size, num_workers=self.val_workers)