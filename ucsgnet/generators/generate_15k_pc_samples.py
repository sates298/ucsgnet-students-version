import numpy as np
import torch
import tqdm

from ucsgnet.common import Evaluation3D
from ucsgnet.flows.shapenet.net_flow3d_cat import FlowNet3d
from ucsgnet.flows.shapenet.reconstruct_3d_shapes import VoxelReconstructor
import argparse
import json
import os
from pathlib import Path
from ucsgnet.utils import convert_to_o3d_mesh
import open3d as o3d


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstructs all shapes in the dataset by predicting values at "
            "each 3D point and then thresholding"
        ),
        add_help=False,
    )

    parser.add_argument(
        "--size", type=int, help="Data size to be used", required=True
    )
    parser.add_argument(
        "--processed",
        dest="processed_data_path",
        type=str,
        help="Base folder of processed data",
        required=True,
    )
    parser.add_argument(
        "--valid",
        dest="valid_file",
        type=str,
        help="Path to valid HDF5 file with the valid data",
        required=True,
    )
    parser.add_argument(
        "--valid_shape_names",
        type=str,
        help=(
            "Path to valid text file with the names for each data point in "
            "the valid dataset"
        ),
        required=True,
    )
    parser.add_argument(
        "--sphere_complexity",
        type=int,
        help="Number of segments lat/lon of the sphere",
        required=False,
        default=16,
    )

    parser.add_argument(
        "--full_resolution_only",
        action="store_true",
        default=True,
        help="Whether use the resolution 64 only instead of whole training",
    )

    parser.add_argument(
        "--flow_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out_folder",
        help="A folder path where pointclouds wil be saved",
        type=str,
        required=True
    )

    parser = FlowNet3d.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    out_dir = Path("data") / args.out_folder

    flow = FlowNet3d.load_from_checkpoint(args.flow_path)
    flow = flow.eval()
    model = flow.net
    model.turn_fine_tuning_mode()
    model.freeze()

    reconstructor = VoxelReconstructor(
        model, args.size, args.sphere_complexity
    )
    with open(
            os.path.join(args.processed_data_path, args.valid_shape_names)
    ) as f:
        file_names = f.read().split("\n")

    with open(os.path.join("data", "shapenet", "taxonomy.json")) as json_f:
        taxonomy = json.loads(json_f.read())

    cat_index_mapper = {}
    cat_name_mapper = {}
    index_cat_mapper = {}
    index_synset_mapper = {}
    counter = 0
    for s_object in taxonomy:
        cat_index_mapper[s_object["synsetId"]] = counter
        cat_name_mapper[s_object["synsetId"]] = s_object["name"]
        index_cat_mapper[counter] = s_object["name"]
        index_synset_mapper[counter] = s_object["synsetId"]
        counter += 1

    num_classes = len(Evaluation3D.CATEGORY_IDS)
    for cat_id in Evaluation3D.CATEGORY_IDS:
        cat_index = Evaluation3D.CATEGORY_IDS.index(cat_id)
        context = torch.zeros((1, num_classes))
        context[0, cat_index] = 1.0
        cat_count = Evaluation3D.CATEGORY_COUNTS[cat_index]
        out_dir_class = out_dir / cat_id
        out_dir_class.mkdir(exist_ok=True, parents=True)

        existing_num = len(os.listdir(out_dir_class))
        # cat_count = cat_count - existing_num
        for iteration_num in tqdm.tqdm(range(existing_num, cat_count)):
            with torch.no_grad():
                try:
                    samples = flow.model.sample(1, context=context).squeeze(0)
                    (
                        pred_reconstructions,
                        points_normals,
                        csg_paths,
                    ) = reconstructor.reconstruct_single_sampled_without_voxels(samples)

                    pred_vertices, pred_triangles = pred_reconstructions[0]
                    mesh = convert_to_o3d_mesh(pred_vertices, pred_triangles)
                    pcd = mesh.sample_points_uniformly(number_of_points=15000)
                    pcd_points = np.asarray(pcd.points)
                    out_dir_file = (out_dir_class / str(iteration_num)).with_suffix(".npy")
                    np.save(out_dir_file, pcd_points)
                except ZeroDivisionError:
                    continue


if __name__ == "__main__":
    main()
