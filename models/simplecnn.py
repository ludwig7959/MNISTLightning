from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import lightning as pl
from lightning.pytorch.utilities.model_summary import ModelSummary

class SimpleCNNForMNIST(pl.LightningModule):
    def __init__(self, optimizer):
        super(SimpleCNNForMNIST, self).__init__()

        self.save_hyperparameters()

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, 32, 3, 1, 1, bias=False)),
            ("bn1", nn.BatchNorm2d(32)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(2)),

            ("conv2", nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ("bn2", nn.BatchNorm2d(64)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(64 * 7 * 7, 128)),
            ("relu1", nn.ReLU(inplace=True)),
            ("dropout1", nn.Dropout(0.5)),
            ("fc2", nn.Linear(128, 10)),
        ]))

        self.optimizer_cfg = optimizer

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
    
if __name__ == "__main__":
    model = SimpleCNNForMNIST()
    model.example_input_array = torch.zeros(1, 1, 28, 28)

    print(ModelSummary(model, max_depth=2))
