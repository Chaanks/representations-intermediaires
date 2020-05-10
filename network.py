import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from dataset import Dataset

class Net(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(self.cfg.input_size, self.cfg.hidden_size)
        self.fc2 = nn.Linear(self.cfg.hidden_size, self.cfg.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = self.fc2(x)
        prob = F.log_softmax(x, dim=1)
        return prob

    def prepare_data(self):
        self.logger.experiment.watch(self, log='all', log_freq=100)

    def train_dataloader(self):
        # data transforms
        # dataset creation
        # return a DataLoader
        self.ds_train = Dataset('./layer{0}/train.embed.layer-{0}'.format(self.cfg.layer))
        train_params = {'batch_size': self.cfg.batch_size,
                        'shuffle': True,
                        'num_workers': 6}

        return DataLoader(self.ds_train, **train_params)

    def val_dataloader(self):
        # can also return a list of val dataloaders
        self.ds_val = Dataset('./layer{0}/dev.embed.layer-{0}'.format(self.cfg.layer))
        val_params = {'batch_size': self.cfg.batch_size,
                        'shuffle': False,
                        'num_workers': 6}

        return DataLoader( self.ds_val, **val_params)

    def test_dataloader(self):
        # can also return a list of test dataloaders
        self.ds_test = Dataset('./layer{0}/test.embed.layer-{0}'.format(self.cfg.layer))
        test_params = {'batch_size': self.cfg.batch_size,
                        'shuffle': False,
                        'num_workers': 6}

        return DataLoader(self.ds_test, **test_params)


    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        logs = {'train_loss': loss}
        #self.logger.log_metrics(logs)

        return {'loss': loss, 'log': logs}

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)

        pred = y_hat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        #logs = {'validation_loss': loss}


        return {'val_loss': loss, 'val_correct': correct}


    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_accuracy = torch.Tensor([x['val_correct'] for x in outputs]).mean()

        logs = {'avg_validation_loss': val_loss_mean, 'val_accuracy': 100. * val_accuracy}

        return {'val_loss': val_loss_mean, 'log': logs}


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)

        pred = y_hat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        #logs = {'test_loss': loss}


        return {'test_loss': loss, 'test_correct': correct}


    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_accuracy = torch.Tensor([x['test_correct'] for x in outputs]).mean()

        logs = {'avg_test_loss': test_loss_mean, 'test_accuracy': 100. * test_accuracy}
        #self.logger.log_metrics(logs)

        return {'test_loss': test_loss_mean, 'log': logs}