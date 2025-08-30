#!/bin/bash

BASE_CMD="python3 defense_models_attack.py --data_dir ./dataset --batch_size 32"

echo "Running attacks..."

# PMAI-FGSM
$BASE_CMD --substitute_model Inc-v3 --target_model hgd \
   --attack pmai_fgsm --eps 0.3 --steps 10 --decay 1.0 --m 5

# APCI-FGSM
$BASE_CMD --substitute_model Inc-v3 --target_model hgd \
   --attack apci_fgsm --eps 0.3 --steps 10 --beta1 0.9 --beta2 0.999 \
   --learning_rate 0.01 --weight_decay 0.004 --epsilon 1e-7 --global_clipnorm 1.0

# Hybrid PMAI-FGSM
$BASE_CMD --substitute_model Inc-v3 --target_model hgd \
   --attack hybrid_pmai_fgsm --eps 0.3 --steps 10 --decay 1.0 --m 5 \
   --use_dim --use_tim --use_sim --resize_rate 1.1 --diversity_prob 0.5 \
   --kernel_size 5 --scale_factors 1.0 0.9 0.8 0.7 0.6

# Hybrid APCI-FGSM
$BASE_CMD --substitute_model Inc-v3 --target_model hgd \
   --attack hybrid_apci_fgsm --eps 0.3 --steps 10 --beta1 0.9 --beta2 0.999 \
   --learning_rate 0.01 --weight_decay 0.004 --epsilon 1e-7 --global_clipnorm 1.0 \
   --use_dim --use_tim --use_sim --resize_rate 1.1 --diversity_prob 0.5 \
   --kernel_size 5 --scale_factors 1.0 0.9 0.8 0.7 0.6
