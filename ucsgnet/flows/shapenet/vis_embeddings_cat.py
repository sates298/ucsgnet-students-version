import argparse
import os.path
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np
import tqdm
from ucsgnet.flows.shapenet.net_flow3d_cat import FlowNet3d
from sklearn.manifold import TSNE

OUT_DIR = Path("paper-stuff/embedding_space/shapenet_cat")
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
        "--flow_path",
        type=str,
        required=True
    )

    return parser.parse_args()


def save_plots(embeddings, categories, path):
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=categories,
        ax=ax,
        palette=sns.color_palette("hls", 13),
        legend='full',
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def create_embeddings(flow, loader):
    codes = []
    samples = []
    categories = []
    for data_index, (embedding, context) in enumerate(tqdm.tqdm(loader)):
        codes.extend(embedding.detach().cpu().numpy())
        category = context.argmax(dim=1)
        for cat in context:
            sample = flow.model.sample(1, context=cat.unsqueeze(0)).squeeze(0)
            samples.extend(sample.detach().cpu().numpy())
        categories.extend(category.detach().cpu().numpy())
    codes = np.asarray(codes)
    samples = np.asarray(samples)
    return codes, samples, categories


def run_umap(codes, samples, categories, split_type):
    reducer = umap.UMAP(random_state=42)
    print(f"[UMAP] Projecting embeddings... - {split_type}")
    embeddings_umap = reducer.fit_transform(codes)
    embeddings_samples_umap = reducer.transform(samples)

    umap_dir = os.path.join(OUT_DIR, "umap", split_type)
    if not os.path.exists(umap_dir):
        os.makedirs(umap_dir, exist_ok=True)
    save_plots(embeddings_umap, categories, os.path.join(umap_dir, "model.png"))
    save_plots(embeddings_samples_umap, categories, os.path.join(umap_dir, "samples.png"))


def run_tsne(codes, samples, categories, split_type):
    print(f"[TSNE] Projecting embeddings... - {split_type}")
    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    embeddings_tsne = tsne.fit_transform(codes)
    embeddings_samples_tsne = tsne.fit_transform(samples)

    tsne_dir = os.path.join(OUT_DIR, "tsne", split_type)
    if not os.path.exists(tsne_dir):
        os.makedirs(tsne_dir, exist_ok=True)

    save_plots(embeddings_tsne, categories, os.path.join(tsne_dir, "model.png"))
    save_plots(embeddings_samples_tsne, categories, os.path.join(tsne_dir, "samples.png"))


def vis_embeddings(args):
    flow = FlowNet3d.load_from_checkpoint(args.flow_path)
    flow.build(args.data_path)
    flow = flow.eval()

    train_loader = flow.train_dataloader()
    val_loader = flow.val_dataloader()

    # train_codes, train_samples, train_categories = create_embeddings(flow, train_loader)
    val_codes, val_samples, val_categories = create_embeddings(flow, val_loader)

    # run_umap(train_codes, train_samples, train_categories, "train")
    run_umap(val_codes, val_samples, val_categories, "val")

    # run_tsne(train_codes, train_samples, train_categories, "train")
    run_tsne(val_codes, val_samples, val_categories, "val")


if __name__ == "__main__":
    vis_embeddings(parse_args())
