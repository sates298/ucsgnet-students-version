import argparse
import typing as t
from typing import Union, Tuple
from typing_extensions import Literal
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from ucsgnet.dataset import HdfsDataset3D, CSVDataset
from ucsgnet.flows.models import SimpleRealNVP
import numpy as np
import os
from ucsgnet.ucsgnet.net_3d import Net


class FlowNet3d(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.train_file_: t.Optional[str] = None
        self.valid_file_: t.Optional[str] = None
        self.data_path_: t.Optional[str] = None
        self.current_data_size_: t.Optional[int] = None

        self.model = SimpleRealNVP(
            features=256,
            hidden_features=hparams.hidden_features,
            context_features=None,
            num_layers=4,
            num_blocks_per_layer=2,
            batch_norm_within_layers=hparams.batch_norm_within_layers,
            batch_norm_between_layers=hparams.batch_norm_between_layers
        )

        self.hparams = hparams;

        (
            trainable_params_count,
            non_trainable_params_count,
        ) = self.num_of_parameters

        self.net = Net.load_from_checkpoint(os.path.join("models",
                                                         "3d_64",
                                                         "initial",
                                                         "ckpts",
                                                         "model.ckpt"
                                                         ))
        self.net = self.net.eval()
        self.net.freeze()

        print("Num of trainable params: {}".format(trainable_params_count))
        print(
            "Num of not trainable params: {}".format(
                non_trainable_params_count
            )
        )

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

    def build(self, data_path: str):
        self.data_path_ = data_path

    def _dataloader(self, training: bool, split_type: Literal["train", "valid"], use_c: bool = False) -> DataLoader:
        # batch_size = self.hparams.batch_size
        # a_file = self.train_file_ if training else self.valid_file_
        # points_to_sample = 16 * 16 * 16
        # if self.current_data_size_ == 64:
        #     points_to_sample *= 4
        # loader = DataLoader(
        #     dataset=HdfsDataset3D(
        #         os.path.join(self.data_path_, a_file),
        #         points_to_sample,
        #         self.hparams.seed,
        #         size=self.current_data_size_,
        #     ),
        #     batch_size=batch_size,
        #     shuffle=training,
        #     drop_last=training,
        #     num_workers=0,
        # )
        # return loader
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

    def forward(self, x, c):
        return self.model.log_prob(inputs=x, context=c)

    def training_step(self, batch, batch_idx):
        self.logger.train()
        # data, _, _, _ = batch
        # with torch.no_grad():
        #     x = self.net.net.encoder_(data)
        x = batch
        loss = -self.model.log_prob(inputs=x).mean()

        tqdm_dict = {
            "train_loss": loss
        }

        logger_dict = {
            "loss": loss
        }

        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": logger_dict}
        )

        return output

    def validation_step(self, batch, batch_idx):
        self.logger.valid()
        # data, _, _, _ = batch
        # with torch.no_grad():
        #     x = self.net.net.encoder_(data)
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

    def validation_end(self, outputs):
        self.logger.valid()
        means = defaultdict(int)
        for output in outputs:
            for key, value in output["log"].items():
                means[key] += value

        means = {key: value / len(outputs) for key, value in means.items()}
        logger_dict = means
        tqdm_dict = {
            "val_" + key: value.item() for key, value in means.items()
        }
        result = {
            "val_loss": means["loss"],
            "progress_bar": tqdm_dict,
            "log": logger_dict,
        }
        return result

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

        parser.add_argument(
            "--use_patience", help="Use Patience", type=bool, default=True
        )

        parser.add_argument(
            "--patience_epochs", help="Patience Epochs", type=int, default=50
        )

        parser.add_argument(
            "--max_epochs",
            type=int,
            help="Maximum number of epochs",
            default=1000,
        )

        parser.add_argument(
            "--batch_norm_between_layers",
            type=bool,
            default=False
        )

        parser.add_argument(
            "--batch_norm_within_layers",
            type=bool,
            default=False
        )

        parser.add_argument(
            "--hidden_features",
            type=int,
            default=100
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Seed for RNG to sample points to predict distance",
        )
        return parser
