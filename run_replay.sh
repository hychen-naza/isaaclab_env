#!/bin/bash
cd /home/hongyi/scalevideomanip/isaacsim_scene && \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python replay.py \
    --pc_data ../output/calibrated_pourtea/pc_data_gravity_aligned.npy \
    --output  ../output/calibrated_pourtea/replay.mp4
