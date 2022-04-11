import argparse

from ucsgnet.dataset import HdfsDataset3DCat
from ucsgnet.ucsgnet.net_3d import Net
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
from ucsgnet.common import SEED


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--weights_path",
        required=True,
        help="Path to the *.ckpt path",
        type=str,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory for metrics",
        required=True,
    )

    parser.add_argument(
        "--full_resolution_only",
        action="store_true",
        default=True,
        help="Whether use the resolution 64 only instead of whole training",
    )

    parser = Net.add_model_specific_args(parser)
    return parser.parse_args()


def generate_embeddings(encoder: torch.nn.Module, dataloader: DataLoader, out_dir: str):
    codes = []
    categories = []
    for image, context, _, _, _ in tqdm(dataloader):
        with torch.no_grad():
            code = encoder(image)
            code = code.detach().cpu().numpy()
            codes.extend(code)
            categories.extend(context.cpu().numpy())

    pd.DataFrame(codes).to_csv(out_dir + ".csv", index=False)
    pd.DataFrame(categories).to_csv(out_dir + "_c.csv", index=False)


def create_loader(training: bool, batch_size: int):
    data_path = os.path.join("data", "hdf5")
    train_path = os.path.join("all_vox256_img_train.hdf5")
    valid_path = os.path.join("all_vox256_img_test.hdf5")
    a_file = train_path if training else valid_path
    points_to_sample = 16 * 16 * 16
    current_data_size_ = 64
    if current_data_size_ == 64:
        points_to_sample *= 4

    loader = DataLoader(
        dataset=HdfsDataset3DCat(
            os.path.join(data_path, a_file),
            points_to_sample,
            SEED,
            size=current_data_size_,
            train=training
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=training,
        num_workers=0
    )
    return loader


def generate_embeddings_dataset(args: argparse.Namespace):
    net = Net.load_from_checkpoint(args.weights_path)
    net = net.eval()
    net.freeze()

    encoder = net.net.encoder_
    train_dataloader = create_loader(True, net.hparams.batch_size)
    # val_dataloader = create_loader(False, net.hparams.batch_size)

    data_dir = args.out_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    generate_embeddings(encoder, train_dataloader, os.path.join(data_dir, "training"))
    # generate_embeddings(encoder, val_dataloader, os.path.join(data_dir, "validation"))


if __name__ == "__main__":
    generate_embeddings_dataset(get_args())
