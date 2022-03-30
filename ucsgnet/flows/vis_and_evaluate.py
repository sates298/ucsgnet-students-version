import argparse
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from ucsgnet.dataset import (
    CADDataset,
    get_simple_2d_transforms,
)

from ucsgnet.ucsgnet.cad.net_cad import Net
from ucsgnet.flows.net_flow import FlowNet
from ucsgnet.ucsgnet.visualize_2d_predicted_shapes import (
    detach_tensor,
    _Sampler,
    visualize_primitives_from_each_layer,
    visualize_reconstruction_path,
    visualize_input_and_output_at_given_layer,
    visualize_combined_first_layer_results_with_primitives,
    visualize_cad_dataset,
)

OUT_DIR = Path("paper-stuff/flow-2d-shapes-visualization-flow14")
OUT_DIR.mkdir(exist_ok=True)

ORIGINAL_WIDTH = 64
ORIGINAL_HEIGHT = 64
SCALE_FACTOR = 8
NUM_PIXELS = ORIGINAL_HEIGHT * ORIGINAL_WIDTH
WAS_USED_COLOR = (0, 255, 0)
SOFTMAX_THRESHOLD = 0.01
LINE_WIDTH = 9

SCALE = max(ORIGINAL_WIDTH * SCALE_FACTOR, ORIGINAL_HEIGHT * SCALE_FACTOR)

COLOR_SAMPLER = _Sampler()


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


def visualize_cad_dataset(model: Net ,flow: FlowNet, h5_file_path:str):
    loader = DataLoader(
        CADDataset(
            h5_file_path,
            data_split="valid",
            transforms=get_simple_2d_transforms(),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for data_index, (image, points, trues, _) in enumerate(tqdm.tqdm(loader)):
        batch_size = image.shape[0]
        samples = flow.model.sample(batch_size)

        (
            output,
            predicted_shapes_distances,
            intermediate_results,
            scaled_distances,
        ) = model.forward_sampled(
            samples,
            points,
            return_distances_to_base_shapes=True,
            return_intermediate_output_csg=True,
            return_scaled_distances_to_shapes=True,
        )

        binarized = model.binarize(output)
        mse_loss = F.mse_loss(binarized, trues).item()
        out_dir = OUT_DIR / "cad" / f"{mse_loss}_{data_index}"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_dir.mkdir(exist_ok=True)

        np_image = (
                image.detach().cpu().numpy().squeeze(axis=(0, 1)) * 255
        ).astype(np.uint8)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)

        cv2.imwrite((out_dir / "ground-truth.png").as_posix(), np_image)

        binarized = (
            binarized.squeeze()
                .detach()
                .cpu()
                .numpy()
                .reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
        )
        np_output = (
            output.squeeze()
                .detach()
                .reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
                .unsqueeze(dim=-1)
                .expand(([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3]))
                .cpu()
                .numpy()
        )
        np_output = (np_output * 255).astype(np.uint8)
        intermediate_prediction = intermediate_results[-1]

        cv2.imwrite(
            (out_dir / "binarized.png").as_posix(),
            (binarized * 255).astype(np.uint8),
        )

        cv2.imwrite(
            (
                    out_dir / f"intermediate_result_"
                              f"{len(model.net.csg_layers_)}_0_used.png"
            ).as_posix(),
            (detach_tensor(intermediate_prediction)[..., 0] * 255)
                .astype(np.uint8)
                .reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH)),
        )

        visualize_primitives_from_each_layer(
            model, np_image, np_output, out_dir
        )
        visualize_input_and_output_at_given_layer(
            intermediate_results[:-1], model, out_dir
        )
        visualize_combined_first_layer_results_with_primitives(
            model, intermediate_results[:-1], out_dir
        )
        visualize_reconstruction_path(model, out_dir, np_image, np_output)


def main(args):
    net = Net.load_from_checkpoint(args.model_path)
    net = net.eval()
    net.freeze()
    net.turn_fine_tuning_mode()

    flow = FlowNet.load_from_checkpoint(args.flow_path)
    flow.build(args.data_path)
    flow = flow.eval()
    # flow.freeze()

    visualize_cad_dataset(net, flow, args.data_path)

if __name__ == "__main__":
    main(parse_args())