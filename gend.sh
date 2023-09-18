#!/bin/bash

python gen_graph.py --feature_path data/features.txt \
                    --data_path data/datasets/train_data.txt \
                    --out_dir data/graphs \
                    --split train \
                    --contact_threshold 5 \
                    --local_radius 12
                    