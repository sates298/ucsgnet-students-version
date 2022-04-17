import argparse
import os
from pathlib import Path

import cv2
import numpy as np

import json

import torch
from tqdm import tqdm

from ucsgnet.flows.shapenet.net_flow3d_cat import FlowNet3d
from ucsgnet.flows.shapenet.reconstruct_3d_shapes import VoxelReconstructor
import argparse
import json
import os
import tempfile
import typing as t
import zipfile
from pathlib import Path
import pandas as pd
import io
import trimesh
import requests

from ucsgnet.generators.generate_edge_data_from_point import get_points, to_tensor
from ucsgnet.utils import write_ply_point_normal, read_point_normal_ply_file
from ucsgnet.visualization.visualize_shapes import xml_head, xml_obj_shapenet, xml_tail, xml_obj, as_mesh, decode_image, \
    render_single_obj


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstructs all shapes in the dataset by predicting values at "
            "each 3D point and then thresholding"
        ),
        add_help=False,
    )

    parser.add_argument(
        "--flow_path",
        type=str,
        required=True
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

    parser = FlowNet3d.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


def generate_edges(
        pc,
        out_dir: str,
):
    out_dir = Path(out_dir)
    pred_vertices, pred_normals = pc[:, :3], pc[:, 3:]

    pred_points = get_points(
        to_tensor(pred_vertices), to_tensor(pred_normals)
    )

    out_file = out_dir / "pred_points.ply"
    write_ply_point_normal(out_file, pred_points)


def main():
    args = get_args()

    out_dir = Path("data") / "3d_shapenet_cat_renders_v2"

    flow = FlowNet3d.load_from_checkpoint(args.flow_path)
    flow = flow.eval()

    with open(
            os.path.join(args.processed_data_path, args.valid_shape_names)
    ) as f:
        file_names = f.read().split("\n")

    model = flow.net
    reconstructor = VoxelReconstructor(
        model, args.size, args.sphere_complexity
    )

    with open(os.path.join("data", "shapenet", "taxonomy.json")) as json_f:
        taxonomy = json.loads(json_f.read())

    cat_index_mapper = {}
    cat_name_mapper = {}
    index_cat_mapper = {}
    counter = 0
    for s_object in taxonomy:
        cat_index_mapper[s_object["synsetId"]] = counter
        cat_name_mapper[s_object["synsetId"]] = s_object["name"]
        index_cat_mapper[counter] = s_object["name"]
        counter += 1

    c_df = pd.read_csv(os.path.join("data", "shapenet_cat_embeddings", "validation_c.csv"))
    mapping_key_i = {key: i for i, key in enumerate(c_df["0"].unique())}
    mapping_i_key = {i: key for i, key in enumerate(c_df["0"].unique())}
    num_classes = len(mapping_key_i)

    num_instances = 3
    for class_index in tqdm(range(1), desc="Classes"):
        for model_index in tqdm(range(num_instances), desc="Instances"):
            original_index = mapping_i_key[class_index]
            class_name = index_cat_mapper[original_index]

            context = torch.zeros((1, num_classes))
            context[0, class_index] = 1.0
            samples = flow.model.sample(1, context=context).squeeze(0)
            (
                pred_reconstsructions,
                points_normals,
                csg_paths,
            ) = reconstructor.reconstruct_single_sampled_without_voxels(samples)

            # pred_mesh.ply : pred_vertices, pred_triangles

            middle_path = str(original_index) + "_" + class_name
            out_tmp_dir = out_dir / middle_path / str(model_index)
            out_tmp_dir.mkdir(parents=True, exist_ok=True)

            pred_vertices, pred_triangles = pred_reconstsructions[0]
            out_pred_mesh_path = out_tmp_dir / "pred_mesh.obj"
            trimesh.Trimesh(pred_vertices, pred_triangles).export(out_pred_mesh_path)

            ply_path = out_tmp_dir / "pred_pc.ply"
            write_ply_point_normal(
                ply_path, points_normals[0]
            )

            generate_edges(points_normals[0], out_tmp_dir)

            csg_path_dir = out_tmp_dir / "csg_path"
            csg_path_dir.mkdir(exist_ok=True, parents=True)
            try:
                for name, mesh in csg_paths[0]:
                    trimesh.Trimesh(mesh.vertices, mesh.faces).export(
                        csg_path_dir / name
                    )
            except ValueError:
                print("Wrong shape")

            try:
                img = render_single_obj(out_pred_mesh_path.as_posix(), False)
                cv2.imwrite(
                    (out_pred_mesh_path.with_suffix(".png")).as_posix(),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )

                # csg_rec_dir = out_tmp_dir / "csg_path_rec"
                # csg_rec_dir.mkdir(exist_ok=True, parents=True)
                # csg_list = list(csg_path_dir.glob("*.obj"))
                # for csg in tqdm(csg_list, desc="CSG components"):
                #     img = render_single_obj(csg.as_posix(), False)
                #     cv2.imwrite(
                #         (csg_rec_dir / csg.with_suffix(".png").name).as_posix(),
                #         cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                #     )
            except TypeError:
                print("Invalid shape {}/{}".format(class_name, str(model_index)))
                continue
            except ValueError:
                print("Invalid shape {}/{}".format(class_name, str(model_index)))
                continue


if __name__ == "__main__":
    main()
