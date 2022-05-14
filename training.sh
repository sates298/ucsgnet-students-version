#!/bin/bash


source /opt/anaconda/bin/activate
conda activate ucsg

python ucsgnet/ucsgnet/cad/train_cad.py \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--experiment_name triangle_unsupervised_v1 \
--lr 0.0001 \
--num_csg_layers 2 \
--out_shapes_per_layer 4 

python ucsgnet/ucsgnet/cad/train_cad.py \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--experiment_name normal_triangle_unsupervised_v2 \
--lr 0.0001 \
--num_csg_layers 2 \
--out_shapes_per_layer 6 

python ucsgnet/ucsgnet/cad/train_cad.py \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--experiment_name normal_triangle_unsupervised_v3 \
--lr 0.0001 \
--num_csg_layers 3 \
--out_shapes_per_layer 4 

python ucsgnet/ucsgnet/cad/train_cad.py \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--experiment_name normal_triangle_unsupervised_v4 \
--lr 0.0001 \
--num_csg_layers 3 \
--out_shapes_per_layer 6 


python ucsgnet/ucsgnet/cad/eval_cad.py \
--weights_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/normal_triangle_unsupervised_v1_main/initial/ckpts/model.ckpt \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--out_dir /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/normal_triangle_unsupervised_v1_main/

python ucsgnet/ucsgnet/cad/eval_cad.py \
--weights_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/normal_triangle_unsupervised_v2_main/initial/ckpts/model.ckpt \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--out_dir /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/normal_triangle_unsupervised_v2_main/

python ucsgnet/ucsgnet/cad/eval_cad.py \
--weights_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/normal_triangle_unsupervised_v3_main/initial/ckpts/model.ckpt \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--out_dir /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/normal_triangle_unsupervised_v3_main/

python ucsgnet/ucsgnet/cad/eval_cad.py \
--weights_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/normal_triangle_unsupervised_v4_main/initial/ckpts/model.ckpt \
--data_path /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/data/cad/cad.h5 \
--out_dir /home/steve/Dokumenty/Studia/project/ucsgnet-students-version/models/normal_triangle_unsupervised_v4_main/
