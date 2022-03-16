import argparse
from typing import Union, Tuple
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing_extensions import Literal
from collections import OrderedDict, defaultdict
from ucsgnet.ucsgnet.net_2d import Net as Net2D
from ucsgnet.dataset import CSVDataset

from nflows.flows.base import Flow
import numpy as np
import os


class FlowNet(pl.LightningModule):
    def __init__(self, model: Flow, hparams: argparse.Namespace):
        super().__init__()
        self.model = model
        self.hparams = hparams

        (
            trainable_params_count,
            non_trainable_params_count,
        ) = self.num_of_parameters

        print("Num of trainable params: {}".format(trainable_params_count))
        print(
            "Num of not trainable params: {}".format(
                non_trainable_params_count
            )
        )

    def build(self, data_path: str, **kwargs):
        self.data_path_ = data_path

    @property
    def num_of_parameters(self) -> Tuple[int, int]:
        total_trainable_params = 0
        total_nontrainable_params = 0

        for param in self.parameters(recurse=True):
            if param.requires_grad:
                total_trainable_params += np.prod(param.shape)
            else:
                total_nontrainable_params += np.prod(param.shape)
        return total_trainable_params, total_nontrainable_params

    def _dataloader(
            self, training: bool, split_type: Literal["train", "valid"], use_c: bool = False
    ) -> DataLoader:
        batch_size = self.hparams.batch_size
        c_path = None
        if split_type == "train":
            x_path = os.path.join(self.data_path_, "training.csv")
            if use_c:
                c_path = os.path.join(self.data_path_, "training_c.csv")
        elif split_type == "valid":
            x_path = os.path.join(self.data_path_, "validation.csv")
            if use_c:
                c_path = os.path.join(self.data_path_, "validation_c.csv")
        else:
            raise Exception("Invalid split type")

        loader = DataLoader(
            dataset=CSVDataset(x_path, c_path),
            batch_size=batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=0,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(True, "train", False)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(False, "valid", False)

    def forward(
            self,
            x: torch.Tensor,
            c: Union[None, torch.Tensor]) -> torch.Tensor:
        return self.model.log_prob(inputs=x, context=c)

    def training_step(self, batch, batch_idx):
        self.logger.train()
        x = batch
        loss = -self.model.log_prob(inputs=x).mean()

        tqdm_dict = {
            "train_loss": loss.item()
        }

        logger_dict = {
            "loss": loss.item()
        }

        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": logger_dict}
        )

        return output

    def validation_step(self, batch, batch_idx):
        self.logger.valid()
        x = batch
        loss = -self.model.log_prob(inputs=x).mean()

        tqdm_dict = {
            "val_loss": loss
        }

        logger_dict = {
            "loss": loss
        }

        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": logger_dict}
        )

        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(0.5, 0.99),
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser])

        parser.add_argument(
            "--lr",
            help="Learning rate of the optimizer",
            type=float,
            default=0.0001,
        )

        parser.add_argument(
            "--batch_size", help="Batch size", type=int, default=32
        )

        return parser
