import argparse
import os.path
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from ucsgnet.dataset import (
    CADDataset,
    get_simple_2d_transforms,
)

from ucsgnet.ucsgnet.cad.net_cad import Net
from ucsgnet.flows.net_flow import FlowNet
from sklearn.manifold import TSNE

OUT_DIR = Path("paper-stuff/embedding_space")
OUT_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=str,
        help="Path to h5 file containing data",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--flow_path",
        type=str,
        required=True
    )

    return parser.parse_args()


def save_plots(embeddings, path):
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        ax=ax,
        legend=None,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def vis_embeddings(args):
    net = Net.load_from_checkpoint(args.model_path)
    net = net.eval()
    net.freeze()
    net.turn_fine_tuning_mode()

    flow = FlowNet.load_from_checkpoint(args.flow_path)
    flow.build(args.data_path)
    flow = flow.eval()

    loader = DataLoader(
        CADDataset(
            args.data_path,
            data_split="valid",
            transforms=get_simple_2d_transforms(),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    encoder = net.net.encoder_
    codes = []
    for data_index, (image, _, _, _) in enumerate(tqdm.tqdm(loader)):
        code = encoder(image)
        codes.extend(code.detach().cpu().numpy())

    codes = np.asarray(codes)
    samples = flow.model.sample(codes.shape[0]).detach().cpu().numpy()

    reducer = umap.UMAP(random_state=42)
    print("[UMAP] Projecting embeddings...")
    embeddings_umap = reducer.fit_transform(codes)
    embeddings_samples_umap = reducer.transform(samples)

    umap_dir = os.path.join(OUT_DIR, "umap")
    if not os.path.exists(umap_dir):
        os.makedirs(umap_dir, exist_ok=True)
    save_plots(embeddings_umap, os.path.join(umap_dir, "model.png"))
    save_plots(embeddings_samples_umap, os.path.join(umap_dir, "samples.png"))

    print("[TSNE] Projecting embeddings...")
    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    embeddings_tsne = tsne.fit_transform(codes)
    embeddings_samples_tsne = tsne.fit_transform(samples)

    tsne_dir = os.path.join(OUT_DIR, "tsne")
    if not os.path.exists(tsne_dir):
        os.makedirs(tsne_dir, exist_ok=True)

    save_plots(embeddings_tsne, os.path.join(tsne_dir, "model.png"))
    save_plots(embeddings_samples_tsne, os.path.join(tsne_dir, "samples.png"))


if __name__ == "__main__":
    vis_embeddings(parse_args())
