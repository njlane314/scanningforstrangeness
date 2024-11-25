#!/bin/bash
cd /gluster/home/niclane/scanningforstrangeness
source /gluster/home/niclane/bin/conda_setup.sh pythondl
source setup.sh
rm class_weights.npy
python3 train.py -c cfg/default.cfg
