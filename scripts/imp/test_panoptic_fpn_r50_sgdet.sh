#!/bin/bash
# sh scripts/imp/test_panoptic_fpn_r50_sgdet.sh

GPU=2
CPU=1
node=73
PORT=29500
jobname=openpsg

# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
  configs/imp/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
  work_dirs/checkpoints/VCTree_epoch_12.pth \
  --eval sgdet

  # --out work_dirs/imp_panoptic_fpn_r50_fpn_1x_sgdet_psg/epoch_12.pkl \