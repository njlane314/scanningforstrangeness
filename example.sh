#!/bin/bash
cd /gluster/home/niclane/scanningforstrangeness
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh 
conda activate pythondl

/gluster/home/niclane/miniforge3/envs/pythondl/bin/python -u segmentation.py --plane 0