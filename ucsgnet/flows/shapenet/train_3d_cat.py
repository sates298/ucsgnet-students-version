import argparse
import json
from ucsgnet.flows.shapenet.net_flow3d_cat import FlowNet3d
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
        required=True
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment",
        default="flow_test",
    )

    parser.add_argument(
        "--pretrained_path",
        dest="checkpoint_path",
        type=str,
        help=(
            "If provided, then it assumes pretraining and continuation of "
            "training"
        ),
        default="",
    )

    parser.add_argument(
        "--full_resolution_only",
        action="store_true",
        default=True,
        help="Whether use the resolution 64 only instead of whole training",
    )

    parser = FlowNet3d.add_model_specific_args(parser)
    return parser.parse_args()


def experiment(args: argparse.Namespace):

    net = FlowNet3d(args)
    net.build(args.data_path)

    if args.checkpoint_path and len(args.checkpoint_path) > 0:
        print(f"Loading pretrained model from: {args.checkpoint_path}")
        net = net.load_from_checkpoint(args.checkpoint_path)

    experiment_name = args.experiment_name
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
        period=2,
    )

    if args.use_patience:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience_epochs,
            verbose=True
        )
    else:
        early_stop_callback = None

    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        distributed_backend="dp",
        default_save_path=model_saving_path,
        logger=logger,
        max_epochs=args.max_epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpointer,
        progress_bar_refresh_rate=1,
    )

    trainer.fit(net)


if __name__ == "__main__":
    experiment(get_args())
