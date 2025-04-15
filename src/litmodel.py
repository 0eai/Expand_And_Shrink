import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-4):

        super().__init__()
        self.channels = 1
        self.width = 28
        self.height = 28
        self.num_classes = 10
        self.hidden_size = 200

        self.model = model
        self.learning_rate = learning_rate
        '''
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels * self.width * self.height, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_classes),
        )
        '''
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(y)
        logits = self(x)
        loss = F.nll_loss(logits, y)
        # preds = torch.argmax(logits, dim=1)
        # acc = accuracy(preds, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        # self.log("train_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", acc, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.encoder(x)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer