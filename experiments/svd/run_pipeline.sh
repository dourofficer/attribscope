#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
RUN="python -m experiments.svd.run_pipeline"
CFG="--config experiments/svd/configs/svd_pipeline.yaml"

$RUN indata $CFG
$RUN cross  $CFG
$RUN self   $CFG