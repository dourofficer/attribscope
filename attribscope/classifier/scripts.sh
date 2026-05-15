# --temperatures 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 \

# MODEL: llama | SEED: 1
CUDA_VISIBLE_DEVICES=1 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models llama-3.1-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings mean last \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 1 \
    --device cuda

# MODEL: llama | SEED: 2
CUDA_VISIBLE_DEVICES=1 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models llama-3.1-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings mean last \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 2 \
    --device cuda

# MODEL: llama | SEED: 3
CUDA_VISIBLE_DEVICES=1 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models llama-3.1-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings mean last \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 3 \
    --device cuda

# MODEL: qwen | SEED: 1
CUDA_VISIBLE_DEVICES=1 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models qwen3-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings mean last \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 1 \
    --device cuda

# MODEL: qwen | SEED: 2
CUDA_VISIBLE_DEVICES=1 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models qwen3-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings mean last \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 2 \
    --device cuda

# MODEL: qwen | SEED: 3
CUDA_VISIBLE_DEVICES=1 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models qwen3-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings mean last \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 3 \
    --device cuda


# ############################################################
# ############################################################
# ############################################################
# ############################################################


# MODEL: llama | SEED: 1
CUDA_VISIBLE_DEVICES=3 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models llama-3.1-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings grad \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 1 \
    --device cuda

# MODEL: llama | SEED: 2
CUDA_VISIBLE_DEVICES=3 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models llama-3.1-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings grad \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 2 \
    --device cuda

# MODEL: llama | SEED: 3
CUDA_VISIBLE_DEVICES=3 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models llama-3.1-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings grad \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 3 \
    --device cuda

# MODEL: qwen | SEED: 1
CUDA_VISIBLE_DEVICES=3 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models qwen3-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings grad \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 1 \
    --device cuda

# MODEL: qwen | SEED: 2
CUDA_VISIBLE_DEVICES=3 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models qwen3-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings grad \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 2 \
    --device cuda

# MODEL: qwen | SEED: 3
CUDA_VISIBLE_DEVICES=3 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models qwen3-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings grad \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.6 2.0 2.4 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.15 0.2 \
    --seeds 3 \
    --device cuda