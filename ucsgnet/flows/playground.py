import os
from torch.utils.data import DataLoader
from ucsgnet.dataset import HdfsDataset3DCat
import h5py
import json
import tqdm

if __name__ == "__main__":

    training = True
    data_path = os.path.join("data", "hdf5")
    train_path = os.path.join("all_vox256_img_train.hdf5")
    train_path_txt = os.path.join("all_vox256_img_train.txt")
    valid_path = os.path.join("all_vox256_img_test.hdf5")
    valid_path_txt = os.path.join("all_vox256_img_test.txt")
    batch_size = 5
    a_file = train_path if training else valid_path
    points_to_sample = 16 * 16 * 16
    current_data_size_ = 64
    if current_data_size_ == 64:
        points_to_sample *= 4

    with open(
        os.path.join(data_path, valid_path_txt)
    ) as f:
        file_names = f.read().split("\n")

    with open(os.path.join("data", "shapenet", "taxonomy.json")) as json_f:
        data = json.loads(json_f.read())

    cat_index_mapper = {}
    cat_name_mapper = {}
    counter = 0
    for s_object in data:
        cat_index_mapper[s_object["synsetId"]] = counter
        cat_name_mapper[s_object["synsetId"]] = s_object["name"]
        counter += 1

    loader = DataLoader(
        dataset=HdfsDataset3DCat(
            os.path.join(data_path, a_file),
            points_to_sample,
            42,
            size=current_data_size_,
            train=training
        ),
        batch_size=batch_size,
        drop_last=training,
        num_workers=0,
        shuffle=False,
    )

    for _, (vox, context, sampled_points, sampled_gt, bounding_vol) in enumerate(tqdm.tqdm(loader)):
        print(vox.shape, ' ', context.shape)
        pass
