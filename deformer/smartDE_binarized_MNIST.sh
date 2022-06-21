#!/usr/bin/env bash

JOB=$(date +%Y%m%d%H%M%S)

echo "train:" >> ${JOB}.yaml
echo "  dataset: mnist" >> ${JOB}.yaml  # "mnist" or "cifar10".
echo "  train_prop: 0.98" >> ${JOB}.yaml
echo "  workers: 10" >> ${JOB}.yaml
echo "  learning_rate: 1.0e-5" >> ${JOB}.yaml
echo "  patience: 5" >> ${JOB}.yaml

echo "model:" >> ${JOB}.yaml
echo "  mlp_layers: [128, 256, 512]" >> ${JOB}.yaml
echo "  nhead: 8" >> ${JOB}.yaml
echo "  dim_feedforward: 2048" >> ${JOB}.yaml
echo "  num_layers: 6" >> ${JOB}.yaml
echo "  dropout: 0.0" >> ${JOB}.yaml

# Save experiment settings.
mkdir -p ${DEFORMER_EXPERIMENTS_DIR}/${JOB}
mv ${JOB}.yaml ${DEFORMER_EXPERIMENTS_DIR}/${JOB}/

gpu=0
cd ${DEFORMER_PROJECT_DIR}
python Smartformer.py ${JOB} ${gpu} # > ${DEFORMER_EXPERIMENTS_DIR}/${JOB}/train.log &
