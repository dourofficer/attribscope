#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
RUN="python -m experiments.reps.run"
CFG="--config experiments/reps/configs/default.yaml"

$RUN grads  $CFG --set grads.loss=kl_temp    --dry-run
$RUN grads  $CFG --set grads.loss=ntp        --dry-run
$RUN grads  $CFG --set grads.loss=kl_uniform --dry-run
$RUN hidden $CFG                             --dry-run