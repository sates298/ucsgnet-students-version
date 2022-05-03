import numpy as np
import tqdm
from ucsgnet.flows.shapenet.net_flow3d_cat import FlowNet3d, Net
from ucsgnet.flows.shapenet.reconstruct_3d_shapes import VoxelReconstructor
import argparse
import json
import os
from pathlib import Path
from ucsgnet.utils import convert_to_o3d_mesh


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
        "--weights_path", required=True, help="Path to the model to load"
    )

    parser = FlowNet3d.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    out_dir = Path("data") / "shapenet_15k_pointclouds"

    model = Net.load_from_checkpoint(args.weights_path)
    model.build("", args.valid_file, args.processed_data_path, 64)
    model.turn_fine_tuning_mode()
    model.freeze()
    model.hparams.batch_size = 1

    reconstructor = VoxelReconstructor(
        model, args.size, args.sphere_complexity
    )
    with open(
            os.path.join(args.processed_data_path, args.valid_shape_names)
    ) as f:
        file_names = f.read().split("\n")

    file_names = [item.strip().split("/") for item in file_names]
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

    loader = model.val_dataloader()
    counter = 0
    for _, batch in enumerate(tqdm.tqdm(loader)):
        voxels = batch[0]
        cat_index = batch[1].detach().numpy()[0]
        synset_index = index_synset_mapper[cat_index]
        gt_rec = reconstructor.reconstruct_single_gt_mesh(voxels)
        vert, triangles = gt_rec[0]
        mesh = convert_to_o3d_mesh(vert, triangles)
        pcd = mesh.sample_points_uniformly(number_of_points=15000)
        pcd_points = np.asarray(pcd.points)
        out_dir_class = out_dir / synset_index
        out_dir_class.mkdir(exist_ok=True, parents=True)
        file_name = file_names[counter][1]
        out_dir_file = (out_dir_class / file_name).with_suffix(".npy")
        np.save(out_dir_file, pcd_points)
        counter += 1


if __name__ == "__main__":
    main()
