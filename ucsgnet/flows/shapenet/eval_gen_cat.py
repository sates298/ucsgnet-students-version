# Code adapted from: https://git.io/Jfeua
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import tqdm
import os

from ucsgnet.generators.generate_edge_data_from_point import get_points, to_tensor


def chamf_dist(
        gt_points: torch.Tensor,
        pred_points: torch.Tensor
):
    gt_num_points = gt_points.shape[0]
    pred_num_points = pred_points.shape[0]

    gt_points = gt_points.squeeze(0)
    pred_points = pred_points.squeeze(0)

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
    # return chamf_dist(ref, sample)


def _pairwise_cd(sample_pcs, ref_pcs, batch_size=16):
    all_cd = []
    sample_pcs = to_tensor(np.asarray(sample_pcs))
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
            # cd = cd_cpu(sample_batch_exp, ref_batch)
            # cd_list.append(cd)
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

    synsetId = args.synsetId

    gt_folder = Path("data/" + args.gt_folder)
    samples_folder = Path("data/" + args.sample_folder)

    gt_pcs = []
    gt_path = gt_folder / synsetId
    for gt_file in os.listdir(gt_path):
        gt_pc = np.load(gt_path / gt_file)
        idx = np.random.randint(gt_pc.shape[0], size=2048)
        gt_pc = gt_pc[idx,:]
        gt_pcs.append(gt_pc)
    gt_pcs = np.asarray(gt_pcs)

    samples_pcs = []
    s_path = samples_folder / synsetId
    for s_file in os.listdir(s_path):
        s_pc = np.load(s_path / s_file)
        idx = np.random.randint(s_pc.shape[0], size=2048)
        s_pc = s_pc[idx, :]
        samples_pcs.append(s_pc)
    samples_pcs = np.asarray(samples_pcs)

    results = {}
    out_folder = Path(args.out_folder)
    pw_cd = _pairwise_cd(samples_pcs, gt_pcs, batch_size=16)
    res_cd = lgan_mmd_cov(pw_cd.t())
    results.update({"%s-CD" % k: v for k, v in res_cd.items()})
    results = {k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in results.items()}

    out_folder.mkdir(exist_ok=True, parents=True)
    out_metrics_path = out_folder / f"{synsetId}-metrics.json"
    with open(out_metrics_path, 'w') as fp:
        json.dump(results, fp)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate standard metrics on 3D autencoding task",
        add_help=False
    )
    parser.add_argument(
        "--synsetId",
        help="Id of shapenet class",
        type=str,
        required=True
    )

    parser.add_argument(
        "--gt_folder",
        help="Ground truth pointclouds",
        type=str,
        required=True
    )

    parser.add_argument(
        "--sample_folder",
        help="Sampled pointclouds",
        type=str,
        required=True
    )

    parser.add_argument(
        "--out_folder",
        help="A folder path where metrics will be saved",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
