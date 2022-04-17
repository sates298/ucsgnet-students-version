# Code adapted from: https://git.io/Jfeua
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tqdm
import pandas as pd
import os

import trimesh

from ucsgnet.common import Evaluation3D
from ucsgnet.flows.shapenet.net_flow3d_cat import FlowNet3d
from ucsgnet.flows.shapenet.reconstruct_3d_shapes import VoxelReconstructor
from ucsgnet.generators.generate_edge_data_from_point import get_points, to_tensor
from ucsgnet.ucsgnet.evaluate_on_3d_data import get_chamfer_distance_and_normal_consistency
from ucsgnet.utils import read_point_normal_ply_file

VERT_KEY = "vertices"
NORM_KEY = "normals"


def get_cd_nc_for_edges(
        gt_folder: Path, class_name: str, object_name: str, pred_points
):
    gt_file = (
            gt_folder / class_name / object_name
    ).with_suffix(".ply")
    gt_vertices, gt_normals = read_point_normal_ply_file(
        gt_file.as_posix()
    )

    gt_points = get_points(
        to_tensor(gt_vertices), to_tensor(gt_normals)
    )

    pred_pc_vertices, pred_pc_normals = pred_points[:, :3], pred_points[:, 3:]
    gt_pc_vertices, gt_pc_normals = gt_points[:, :3], gt_points[:, 3:]

    return get_chamfer_distance_and_normal_consistency(
        to_tensor(gt_pc_vertices),
        to_tensor(pred_pc_vertices),
        to_tensor(gt_pc_vertices),
        to_tensor(pred_pc_normals),
    )


def get_cd_nc_for_points(
        ground_truth_point_surface: Path,
        class_name: str,
        object_name: str,
        pred_pc,
) -> Tuple[float, float]:
    # read ground truth point cloud
    gt_file = (ground_truth_point_surface / class_name / object_name).with_suffix(
        ".ply"
    )
    gt_pc_vertices, gt_pc_normals = read_point_normal_ply_file(gt_file.as_posix())

    # read preds
    pred_pc_vertices, pred_pc_normals = pred_pc[:, :3], pred_pc[:, 3:]

    # compute Chamfer distance and Normal consistency
    return get_chamfer_distance_and_normal_consistency(
        to_tensor(gt_pc_vertices),
        to_tensor(pred_pc_vertices),
        to_tensor(gt_pc_vertices),
        to_tensor(pred_pc_normals),
    )


def chamfer_dist(gt_points, pred_points):
    gt_num_points = gt_points.shape[0]
    pred_num_points = pred_points.shape[0]

    points_gt_matrix = gt_points.unsqueeze(1).expand(
        [gt_points.shape[0], pred_num_points, gt_points.shape[-1]]
    )
    points_pred_matrix = pred_points.unsqueeze(0).expand(
        [gt_num_points, pred_points.shape[0], pred_points.shape[-1]]
    )
    distances = (points_gt_matrix - points_pred_matrix).pow(2).sum(dim=-1)
    match_pred_gt = distances.argmin(dim=0)
    match_gt_pred = distances.argmin(dim=1)

    dist_pred_gt = (pred_points - gt_points[match_pred_gt]).pow(2).sum(dim=-1).mean()
    dist_gt_pred = (gt_points - pred_points[match_gt_pred]).pow(2).sum(dim=-1).mean()

    chamfer_distance = dist_pred_gt + dist_gt_pred
    return chamfer_distance.item()


def cd_cpu(sample, ref):
    x, y = sample, ref
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(sample).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def _pairwise_cd(samle_pcs, ref_pcs, batch_size=16):
    all_cd = []
    sample_pcs = to_tensor(np.asarray(samle_pcs))
    ref_pcs = to_tensor(np.asarray(ref_pcs))
    num_samples = sample_pcs.shape[0]
    num_refs = ref_pcs.shape[0]
    ref_pcs = ref_pcs.contiguous()
    for sample_b_start in tqdm.tqdm(range(num_samples), total=num_samples):
        sample_batch = sample_pcs[sample_b_start]
        cd_list = list()
        for ref_b_start in range(0, num_refs, batch_size):
            ref_b_end = min(num_refs, ref_b_start + batch_size)
            ref_batch = to_tensor(np.asarray(ref_pcs[ref_b_start:ref_b_end]))
            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()
            dl, dr = cd_cpu(sample_batch_exp, ref_batch)
            cd_list.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))
        cd_list = torch.cat(cd_list, dim=1)
        all_cd.append(cd_list)
    all_cd = torch.cat(all_cd, dim=0)
    return all_cd


def lgan_mmd_cov(all_dist):
    n_sample, n_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(n_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def main():
    args = get_args()
    with open(args.valid_shape_names_file, "r") as f:
        eval_list = f.readlines()
    eval_list = [item.strip().split("/") for item in eval_list]
    print(f"Num objects: {len(eval_list)}")

    eval_categories = {}
    for eval_item in eval_list:
        class_name, instance_name = eval_item[0], eval_item[1]
        if class_name in eval_categories:
            eval_categories[class_name].append(instance_name)
        else:
            eval_categories[class_name] = [instance_name]

    flow = FlowNet3d.load_from_checkpoint(args.flow_path)
    flow = flow.eval()
    model = flow.net
    reconstructor = VoxelReconstructor(
        model, args.size, args.sphere_complexity
    )

    reconstructed_shapes_folder = Path(args.rec_folder)
    ground_truth_point_surface = Path(args.ground_truth_point_surface)
    raw_shapenet_data_folder = Path(args.raw_shapenet_data_folder)

    out_per_obj = defaultdict(dict)
    category_chamfer_distance_sum = defaultdict(float)
    category_normal_consistency_sum = defaultdict(float)

    edge_category_chamfer_distance_sum = defaultdict(float)
    edge_category_normal_consistency_sum = defaultdict(float)

    category_count = defaultdict(int)
    mean_metrics = defaultdict(float)
    total_entries = 0

    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    errors = []

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

    results = {}

    gt_pointclouds = {}
    pred_pointclouds = {}
    for cat_id in Evaluation3D.CATEGORY_IDS[:1]:
        gt_pointclouds[cat_id] = {}
        gt_pointclouds[cat_id][VERT_KEY] = []
        gt_pointclouds[cat_id][NORM_KEY] = []
        for instance in tqdm.tqdm(eval_categories[cat_id]):
            gt_file = (ground_truth_point_surface / cat_id / instance).with_suffix(
                ".ply"
            )
            gt_pc_vertices, gt_pc_normals = read_point_normal_ply_file(gt_file.as_posix())
            gt_pointclouds[cat_id][VERT_KEY].append(gt_pc_vertices)
            gt_pointclouds[cat_id][NORM_KEY].append(gt_pc_normals)

        number_refs = len(gt_pointclouds[cat_id][VERT_KEY])
        pred_pointclouds[cat_id] = {}
        pred_pointclouds[cat_id][VERT_KEY] = []
        pred_pointclouds[cat_id][NORM_KEY] = []
        context = torch.zeros((1, num_classes))
        cat_index = Evaluation3D.CATEGORY_IDS.index(cat_id)
        context[0, cat_index] = 1.0

        with torch.no_grad():
            for _ in tqdm.tqdm(range(number_refs)):
                try:
                    samples = flow.model.sample(1, context=context).squeeze(0)
                    (
                        pred_reconstsructions,
                        points_normals,
                        csg_paths,
                    ) = reconstructor.reconstruct_single_sampled_without_voxels(samples)

                    pc_vert, pc_normals = points_normals[0][:, :3], points_normals[0][:, 3:]
                    pred_pointclouds[cat_id][VERT_KEY].append(pc_vert)
                    pred_pointclouds[cat_id][NORM_KEY].append(pc_normals)
                except ZeroDivisionError:
                    continue
        pw_cd = _pairwise_cd(
            pred_pointclouds[cat_id][VERT_KEY],
            gt_pointclouds[cat_id][VERT_KEY],
            batch_size=12
        )

        res_cd = lgan_mmd_cov(pw_cd.t())
        results.update({"%s-CD" % k: v for k, v in res_cd.items()})
        results = {k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in results.items()}

        out_metrics_path = out_folder / f"{cat_id}-metrics.json"

        with open(out_metrics_path, 'w') as fp:
            json.dump(results, fp)

    # with tqdm.trange(len(eval_list)) as pbar:
    #     for idx in pbar:
    #         class_name = eval_list[idx][0]
    #         object_name = eval_list[idx][1]
    #         pbar.set_postfix_str(f"{class_name}/{object_name}")
    #
    #         cat_id = Evaluation3D.CATEGORY_IDS.index(class_name)
    #         context = torch.zeros((1, num_classes))
    #         context[0, cat_id] = 1.0
    #         samples = flow.model.sample(1, context=context).squeeze(0)
    #         (
    #             pred_reconstsructions,
    #             points_normals,
    #             csg_paths,
    #         ) = reconstructor.reconstruct_single_sampled_without_voxels(samples)
    #
    #         pc_vert, pc_normals = points_normals[0][:, :3], points_normals[0][:, 3:]
    #         pred_points = get_points(
    #             to_tensor(pc_vert), to_tensor(pc_normals)
    #         )
    #
    #         try:
    #             edge_cd, edge_nc = get_cd_nc_for_edges(
    #                 ground_truth_point_surface,
    #                 class_name,
    #                 object_name,
    #                 pred_points
    #             )
    #         except RuntimeError:
    #             errors.append((class_name, object_name))
    #             continue
    #
    #         edge_category_chamfer_distance_sum[cat_id] += edge_cd
    #         edge_category_normal_consistency_sum[cat_id] += edge_nc
    #
    #         out_per_obj[class_name][object_name] = {}
    #         out_per_obj[class_name][object_name]["chamfer_distance_edge"] = edge_cd
    #         out_per_obj[class_name][object_name]["normal_consistency_edge"] = edge_nc
    #
    #         mean_metrics["chamfer_distance_edge"] += edge_cd
    #         mean_metrics["normal_consistency_edge"] += edge_nc
    #
    #         # cd and nc
    #         points_cd, points_nc = get_cd_nc_for_points(
    #             ground_truth_point_surface,
    #             class_name,
    #             object_name,
    #             points_normals[0]
    #         )
    #
    #         category_chamfer_distance_sum[cat_id] += points_cd
    #         category_normal_consistency_sum[cat_id] += points_nc
    #
    #         out_per_obj[class_name][object_name] = {
    #             "id": cat_id,
    #             "chamfer_distance": points_cd,
    #             "normal_consistency": points_nc,
    #         }
    #
    #         mean_metrics["chamfer_distance"] += points_cd
    #         mean_metrics["normal_consistency"] += points_nc
    #
    #         category_count[cat_id] += 1
    #         total_entries += 1
    #
    # print(f"{len(errors)} error shapes")
    #
    # per_category = {
    #     Evaluation3D.CATEGORY_NAMES[cat_id]: {
    #         "chamfer_distance": category_chamfer_distance_sum[cat_id]
    #                             / category_count[cat_id],
    #         "normal_consistency": category_normal_consistency_sum[cat_id]
    #                               / category_count[cat_id],
    #         "chamfer_distance_edge": edge_category_chamfer_distance_sum[cat_id]
    #                                  / category_count[cat_id],
    #         "normal_consistency_edge": edge_category_normal_consistency_sum[cat_id]
    #                                    / category_count[cat_id],
    #     }
    #     for cat_id in category_chamfer_distance_sum.keys()
    # }
    #
    # with open(out_folder / "per_category_metrics.json", "w") as f:
    #     json.dump(per_category, f, indent=4)
    #
    # with open(out_folder / "per_object_metrics.json", "w") as f:
    #     json.dump(out_per_obj, f, indent=4)
    #
    # with open(out_folder / "errors.txt", "w") as f:
    #     f.write("\n".join(comp[0] + "/" + comp[1] for comp in errors))
    #
    # mean_metrics = {
    #     name: metric / total_entries for name, metric in mean_metrics.items()
    # }
    # with open(out_folder / "mean_metrics.json", "w") as f:
    #     json.dump(mean_metrics, f, indent=4)
    # print(mean_metrics)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate standard metrics on 3D autencoding task",
        add_help=False
    )
    parser.add_argument(
        "--flow_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--valid_shape_names_file",
        help="A file path containing names of shapes to validate on",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--rec_folder",
        help="A folder path containing reconstructed shapes and point clouds",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--raw_shapenet_data_folder",
        help="A folder path to ground truth ShapeNet shapes",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ground_truth_point_surface",
        help="A folder path to ground truth sampled points of ShapeNet shapes",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out_folder",
        help="A folder path where metrics will be saved",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--size", type=int, help="Data size to be used", required=True
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


if __name__ == "__main__":
    main()
