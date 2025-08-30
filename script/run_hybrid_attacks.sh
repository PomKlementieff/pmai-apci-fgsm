#!/bin/bash
BASE_CMD="python3 blackbox_attack_runner.py --substitute_model Inc-v3 --target_model Res-101 --data_dir ./dataset --batch_size 32"

# Basic settings for all attacks
ATTACK_BASE="$BASE_CMD --attack hybrid_fgsm --eps 0.3"

# 1. DIM only
echo "Running DIM only..."
$ATTACK_BASE \
    --use_dim \
    --resize_rate 1.1 --diversity_prob 0.5

# 2. TIM only
echo "Running TIM only..."
$ATTACK_BASE \
    --use_tim \
    --kernel_size 5

# 3. SIM only
echo "Running SIM only..."
$ATTACK_BASE \
    --use_sim \
    --scale_factors 1.0 0.9 0.8 0.7 0.6

# 4. PIM only
echo "Running PIM only..."
$ATTACK_BASE \
    --use_pim \
    --patch_size 16

# 5. DIM + TIM
echo "Running DIM + TIM..."
$ATTACK_BASE \
    --use_dim --use_tim \
    --resize_rate 1.1 --diversity_prob 0.5 \
    --kernel_size 5

# 6. DIM + SIM
echo "Running DIM + SIM..."
$ATTACK_BASE \
    --use_dim --use_sim \
    --resize_rate 1.1 --diversity_prob 0.5 \
    --scale_factors 1.0 0.9 0.8 0.7 0.6

# 7. TIM + SIM
echo "Running TIM + SIM..."
$ATTACK_BASE \
    --use_tim --use_sim \
    --kernel_size 5 \
    --scale_factors 1.0 0.9 0.8 0.7 0.6

# 8. DIM + TIM + SIM
echo "Running DIM + TIM + SIM..."
$ATTACK_BASE \
    --use_dim --use_tim --use_sim \
    --resize_rate 1.1 --diversity_prob 0.5 \
    --kernel_size 5 \
    --scale_factors 1.0 0.9 0.8 0.7 0.6
