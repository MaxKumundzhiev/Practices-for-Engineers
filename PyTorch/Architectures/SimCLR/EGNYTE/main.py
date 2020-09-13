# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import torch
import pytorch_lightning as pl

from argparse import Namespace

from common.embeddings_net import ImageEmbedding
from common.augumentation import PretrainingDatasetWrapper, stl10_unlabeled
from common.vectorized_contrastive_loss import ContrastiveLoss


from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler

from torch.optim import RMSprop
from torch.multiprocessing import cpu_count
from pytorch_lightning.loggers import WandbLogger


class ImageEmbeddingModule (pl.LightningModule):
    def __init__(self, hparams):
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        super().__init__()
        self.hparams = hparams
        self.model = ImageEmbedding()
        self.loss = ContrastiveLoss(hparams.batch_size)

    def total_steps(self):
        return len (self.train_dataloader()) // self.hparams.epochs

    def train_dataloader(self):
        return DataLoader (PretrainingDatasetWrapper(stl10_unlabeled,
                                                      debug=getattr(self.hparams, "debug", False)),
                           batch_size=self.hparams.batch_size,
                           num_workers=cpu_count(),
                           sampler=SubsetRandomSampler(list(range(hparams.train_size))),
                           drop_last=True)

    def val_dataloader(self):
        return DataLoader (PretrainingDatasetWrapper (stl10_unlabeled,
                                                      debug=getattr (self.hparams, "debug", False)),
                           batch_size=self.hparams.batch_size,
                           shuffle=False,
                           num_workers=cpu_count (),
                           sampler=SequentialSampler (
                               list (range (hparams.train_size + 1, hparams.train_size + hparams.validation_size))),
                           drop_last=True)

    def forward(self, X):
        return self.model (X)

    def step(self, batch, step_name="train"):
        (X, Y), y = batch
        embX, projectionX = self.forward (X)
        embY, projectionY = self.forward (Y)
        loss = self.loss (projectionX, projectionY)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}

    def training_step(self, batch, batch_idx):
        return self.step (batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step (batch, "val")

    def validation_end(self, outputs):
        if len (outputs) == 0:
            return {"val_loss": torch.tensor (0)}
        else:
            loss = torch.stack ([x["val_loss"] for x in outputs]).mean ()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def configure_optimizers(self):
        optimizer = RMSprop (self.model.parameters (), lr=self.hparams.lr)
        return [optimizer], []


if __name__ == "__main__":
    hparams = Namespace(
        lr=1e-3,
        epochs=50,
        batch_size=160,
        train_size=10000,
        validation_size=1000
    )

    module = ImageEmbeddingModule(hparams)
    t = pl.Trainer(gpus=1)
    lr_finder = t.lr_find(module)
