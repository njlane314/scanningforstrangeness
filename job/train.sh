#!/bin/bash
cd /gluster/home/niclane/scanningforstrangeness
source /gluster/home/niclane/bin/conda_setup.sh pythondl
source setup.sh

/gluster/home/niclane/miniforge3/envs/pythondl/bin/python train.py 