import argparse
from ucsgnet.ucsgnet.cad.net_cad import Net
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=str,
        help="Path to h5 file containing data",
        required=True,
    )

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
    return parser.parse_args()


def generate_embeddings(encoder: torch.nn.Module, dataloader: DataLoader, out_dir: str):
    codes = []
    for image, _, _, _ in tqdm(dataloader):
        with torch.no_grad():
            code = encoder(image)
            code = code.detach().cpu().numpy()
            codes.extend(code)

    pd.DataFrame(codes).to_csv(out_dir, index=False)


def generate_embeddings_dataset(args: argparse.Namespace):
    net = Net.load_from_checkpoint(args.weights_path)
    net.build(args.data_path)
    net = net.eval()
    net.freeze()

    encoder = net.net.encoder_
    train_dataloader = net.train_dataloader()
    val_dataloader = net.val_dataloader()

    data_dir = args.out_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    generate_embeddings(encoder, train_dataloader, os.path.join(data_dir, "training3.csv"))
    generate_embeddings(encoder, val_dataloader, os.path.join(data_dir, "validation3.csv"))


if __name__ == "__main__":
    generate_embeddings_dataset(get_args())
