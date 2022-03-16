import argparse
import json
from ucsgnet.flows.net_flow import FlowNet
from ucsgnet.flows.models import SimpleRealNVP
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer

from ucsgnet.callbacks import ModelCheckpoint
from ucsgnet.loggers import TensorBoardLogger


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=str,
        help="Path to h5 file containing data",
        required=True,
    )

    parser = FlowNet.add_model_specific_args(parser)
    return parser.parse_args()


def experiment(args: argparse.Namespace):
    flow = SimpleRealNVP(256, 100, None)
    net = FlowNet(flow, args)
    net.build(args.data_path)

    experiment_name = "test_flow"
    model_saving_path = os.path.join("models", experiment_name, "initial")

    if not os.path.exists(model_saving_path):
        os.makedirs(model_saving_path, exist_ok=True)

    with open(os.path.join(model_saving_path, "params.json"), "w") as f:
        json.dump(vars(args), f)
    logger = TensorBoardLogger(
        os.path.join(model_saving_path, "logs"), log_train_every_n_step=200
    )

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(model_saving_path, "ckpts", "model.ckpt"),
        monitor="val_loss",
        period=10,
    )

    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        distributed_backend="dp",
        default_save_path=model_saving_path,
        logger=logger,
        max_epochs=250,
        early_stop_callback=None,
        checkpoint_callback=checkpointer,
        progress_bar_refresh_rate=1,
    )

    trainer.fit(net)


if __name__ == "__main__":
    experiment(get_args())
