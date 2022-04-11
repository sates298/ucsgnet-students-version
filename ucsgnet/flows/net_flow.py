import argparse
from typing import Union, Tuple
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing_extensions import Literal
from collections import OrderedDict, defaultdict
from ucsgnet.ucsgnet.net_2d import Net as Net2D
from ucsgnet.dataset import CSVDataset, get_simple_2d_transforms, CADDataset
from ucsgnet.flows.models import SimpleRealNVP
from nflows.flows.base import Flow
import numpy as np
import os
from ucsgnet.ucsgnet.cad.net_cad import Net


class FlowNet(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.model = SimpleRealNVP(
            features=256,
            hidden_features=hparams.hidden_features,
            context_features=None,
            num_layers=4,
            num_blocks_per_layer=2,
            batch_norm_between_layers=hparams.batch_norm_between_layers,
            batch_norm_within_layers=hparams.batch_norm_within_layers
        )
        self.hparams = hparams

        (
            trainable_params_count,
            non_trainable_params_count,
        ) = self.num_of_parameters

        self.net = Net.load_from_checkpoint(os.path.join("models",
                                                    "cad_main",
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
            x_path = os.path.join(self.data_path_, "training2.csv")
            if use_c:
                c_path = os.path.join(self.data_path_, "training_c.csv")
        elif split_type == "valid":
            x_path = os.path.join(self.data_path_, "validation2.csv")
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
        # batch_size = self.hparams.batch_size
        # transforms = get_simple_2d_transforms()
        # loader = DataLoader(
        #     dataset=CADDataset(self.data_path_, split_type, transforms),
        #     batch_size=batch_size,
        #     shuffle=training,
        #     drop_last=training,
        #     num_workers=0,
        # )
        # return loader

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

        # image, points, trues, bounding_volume = batch
        # with torch.no_grad():
        #     x = self.net.net.encoder_(image)

        # x = x + self.params['noise'] * torch.rand_like(x)
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
        x = batch
        # image, points, trues, bounding_volume = batch
        # with torch.no_grad():
        #     x = self.net.net.encoder_(image)

        # x = x + self.params['noise'] * torch.rand_like(x)
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
        return parser

    # def __next_elem_from_loader(self, loader: DataLoader):
    #     x = next(iter(loader))
    #     if self.on_gpu:
    #         x = x.cuda()
    #     return x
    #
