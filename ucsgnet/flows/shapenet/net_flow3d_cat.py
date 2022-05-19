import argparse
import typing as t
from typing import Union, Tuple
from typing_extensions import Literal
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from ucsgnet.dataset import HdfsDataset3DCat, CSVDataset
from ucsgnet.flows.models import SimpleRealNVP, MaskedAutoregressiveFlow
import numpy as np
import os
from ucsgnet.ucsgnet.net_3d import Net
from sklearn.metrics import accuracy_score


class FlowNet3d(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace, use_autoregressive=False):
        super().__init__()
        self.train_file_: t.Optional[str] = None
        self.valid_file_: t.Optional[str] = None
        self.data_path_: t.Optional[str] = None
        self.current_data_size_: t.Optional[int] = None
        self.num_classes = 13

        if use_autoregressive:
            self.model = MaskedAutoregressiveFlow(
                features=256,
                hidden_features=hparams.hidden_features,
                context_features=self.num_classes,
                num_layers=5,
                num_blocks_per_layer=2,
                batch_norm_between_layers=hparams.batch_norm_between_layers,
                batch_norm_within_layers=hparams.batch_norm_within_layers,
                use_random_permutations=True
            )
        else:
            self.model = SimpleRealNVP(
                features=256,
                hidden_features=hparams.hidden_features,
                context_features=self.num_classes,
                num_layers=5,
                num_blocks_per_layer=2,
                batch_norm_within_layers=hparams.batch_norm_within_layers,
                batch_norm_between_layers=hparams.batch_norm_between_layers
            )

        self.hparams = hparams

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

        print("[FLOW] Num of trainable params: {}".format(trainable_params_count))
        print(
            "[FLOW] Num of not trainable params: {}".format(
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
            dataset=CSVDataset(x_path, c_path, c_num=self.num_classes),
            batch_size=batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=0,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(True, "train", True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(False, "valid", True)

    def forward(self, x, c):
        return self.model.log_prob(inputs=x, context=c)

    def predict(self, flow, x, num_classes, log_weights=None):
        results = []
        with torch.no_grad():
            for i in range(num_classes):
                context = torch.zeros(len(x), num_classes)
                if torch.cuda.is_available():
                    context = context.cuda()
                context[:, i] = 1.0
                results.append(flow.log_prob(x, context).detach().cpu().numpy())

        y_prob = np.stack(results, axis=1)
        if log_weights is not None:
            y_prob = y_prob + log_weights
        y_hat = y_prob.argmax(axis=1)
        return y_hat

    def training_step(self, batch, batch_idx):
        self.logger.train()
        x, y = batch

        x = x + self.hparams.noise * torch.rand_like(x)
        loss = -self.model.log_prob(inputs=x, context=y).mean()

        y_hat = self.predict(self.model, x, num_classes=self.num_classes)

        acc = accuracy_score(y.detach().cpu().numpy().argmax(axis=1), y_hat)

        tqdm_dict = {
            "train_loss": loss,
            "train_acc": acc
        }

        logger_dict = {
            "loss": loss,
            "acc": acc
        }

        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": logger_dict,
                "y": y.detach().cpu().numpy().argmax(axis=1).tolist(),
                "y_hat": y_hat.tolist()
            }
        )

        return output

    def training_epoch_end(self, outputs):
        self.logger.train()
        y = np.hstack([el["y"] for el in outputs])
        y_hat = np.hstack([el["y_hat"] for el in outputs])
        means = defaultdict(int)
        for output in outputs:
            for key, value in output["log"].items():
                means[key] += value

        means = {key: value / len(outputs) for key, value in means.items()}
        logger_dict = means
        acc = accuracy_score(y, y_hat)

        tmp = {"train_" + key: value.item() for key, value in means.items()}

        tqdm_dict = {
            "train_acc": acc,
            list(tmp.keys())[0]: list(tmp.values())[0],
        }

        logger_dict["train_acc"] = acc

        result = {
            "train_loss": means["loss"],
            "progress_bar": tqdm_dict,
            "log": logger_dict,
            "train_acc": accuracy_score(y, y_hat)
        }
        return result

    def validation_step(self, batch, batch_idx):
        self.logger.valid()
        x, y = batch
        loss = -self.model.log_prob(inputs=x, context=y).mean()

        y_hat = self.predict(self.model, x, self.num_classes)

        acc = accuracy_score(y.detach().cpu().numpy().argmax(axis=1), y_hat)
        tqdm_dict = {
            "val_loss": loss,
            "val_acc": acc
        }

        logger_dict = {
            "loss": loss,
            "acc": acc
        }

        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": logger_dict,
                "y": y.detach().cpu().numpy().argmax(axis=1).tolist(),
                "y_hat": y_hat.tolist()
            }
        )

        return output

    def validation_epoch_end(self, outputs):
        self.logger.valid()
        y = np.hstack([el["y"] for el in outputs])
        y_hat = np.hstack([el["y_hat"] for el in outputs])
        means = defaultdict(int)
        for output in outputs:
            for key, value in output["log"].items():
                means[key] += value

        means = {key: value / len(outputs) for key, value in means.items()}
        logger_dict = means
        acc = accuracy_score(y, y_hat)

        tmp = {"val_" + key: value.item() for key, value in means.items()}

        tqdm_dict = {
            "val_acc": acc,
            list(tmp.keys())[0]: list(tmp.values())[0],
        }

        logger_dict["val_acc"] = acc

        result = {
            "val_loss": means["loss"],
            "progress_bar": tqdm_dict,
            "log": logger_dict,
            "val_acc": accuracy_score(y, y_hat)
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
            default=220
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Seed for RNG to sample points to predict distance",
        )

        parser.add_argument(
            "--noise",
            type=float,
            default=0.001
        )
        return parser


class FlowNet3dMAF(FlowNet3d):
    def forward(self, x, c):
        return self.model.log_prob(inputs=x, context=c)

    def __init__(self, hparams: argparse.Namespace):
        super().__init__(hparams, use_autoregressive=True)
