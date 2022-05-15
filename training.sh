#!/bin/bash


source /opt/anaconda/bin/activate
conda activate ucsg

# python ucsgnet/ucsgnet/cad/train_cad.py \
# --data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
# --labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
# --experiment_name triangle_supervised_v1 \
# --lr 0.0001 \
# --num_csg_layers 2 \
# --out_shapes_per_layer 4 

# python ucsgnet/ucsgnet/cad/train_cad.py \
# --data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
# --labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
# --experiment_name triangle_supervised_v2 \
# --lr 0.0001 \
# --num_csg_layers 2 \
# --out_shapes_per_layer 6 

# python ucsgnet/ucsgnet/cad/train_cad.py \
# --data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
# --labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
# --experiment_name triangle_supervised_v3 \
# --lr 0.0001 \
# --num_csg_layers 3 \
# --out_shapes_per_layer 4 

# python ucsgnet/ucsgnet/cad/train_cad.py \
# --data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
# --labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
# --experiment_name triangle_supervised_v4 \
# --lr 0.0001 \
# --num_csg_layers 3 \
# --out_shapes_per_layer 6 

# python ucsgnet/ucsgnet/cad/train_cad.py \
# --data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
# --labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
# --experiment_name triangle_supervised_v5 \
# --lr 0.0001 \
# --num_csg_layers 3 \
# --out_shapes_per_layer 8 


python ucsgnet/ucsgnet/cad/eval_cad.py \
--weights_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v1_main/initial/ckpts/model.ckpt \
--labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--out_dir /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v1_main/

python ucsgnet/ucsgnet/cad/eval_cad.py \
--weights_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v2_main/initial/ckpts/model.ckpt \
--labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--out_dir /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v2_main/

python ucsgnet/ucsgnet/cad/eval_cad.py \
--weights_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v3_main/initial/ckpts/model.ckpt \
--labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--out_dir /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v3_main/

python ucsgnet/ucsgnet/cad/eval_cad.py \
--weights_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v4_main/initial/ckpts/model.ckpt \
--labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--out_dir /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v4_main/

python ucsgnet/ucsgnet/cad/eval_cad.py \
--weights_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v5_main/initial/ckpts/model.ckpt \
--labels_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad_labels_8.h5 \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--out_dir /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/triangle_supervised_v5_main/
