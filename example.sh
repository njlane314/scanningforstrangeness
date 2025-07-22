#!/bin/bash
cd /gluster/home/niclane/scanningforstrangeness
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh 
conda activate contrastive

/gluster/data/dune/niclane/miniforge/envs/contrastive/bin/python -u contrastive.py
