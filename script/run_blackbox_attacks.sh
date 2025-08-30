#!/bin/bash

BASE_CMD="python3 blackbox_attack_runner.py --substitute_model Con-B --target_model Con-B --data_dir ./dataset --batch_size 32"

# FGSM Attack
#$BASE_CMD --attack fgsm --eps 0.3

# I-FGSM Attack
#$BASE_CMD --attack i_fgsm --eps 0.3 --steps 10

# MI-FGSM Attack
#$BASE_CMD --attack mi_fgsm --eps 0.3 --steps 10 --decay 1.0

# NI-FGSM Attack
#$BASE_CMD --attack ni_fgsm --eps 0.3 --steps 10 --decay 1.0

# PI-FGSM Attack
#$BASE_CMD --attack pi_fgsm --eps 0.3 --steps 10 --decay 1.0

# EMI-FGSM Attack
#$BASE_CMD --attack emi_fgsm --eps 0.3 --steps 10 --decay 1.0 --N 11 --eta 7

# SNI-FGSM Attack
#$BASE_CMD --attack sni_fgsm --eps 0.3 --steps 10 --decay 1.0

# BNI-FGSM Attack
#$BASE_CMD --attack bni_fgsm --eps 0.3 --steps 10 --decay 1.0

# DNI-FGSM Attack
#$BASE_CMD --attack dni_fgsm --eps 0.3 --steps 10 --decay 1.0

# QHMI-FGSM Attack
#$BASE_CMD --attack qhmi_fgsm --eps 0.3 --steps 10 --decay 1.0 --nu 0.7

# ANAGI-FGSM Attack
#$BASE_CMD --attack anagi_fgsm --eps 0.3 --steps 10 --decay 1.0

# AI-FGSM Attack
#$BASE_CMD --attack ai_fgsm --eps 0.3 --steps 20 --beta1 0.9 --beta2 0.999 --learning_rate 0.005 --epsilon 1e-8 --weight_decay 0.0005 --amsgrad --lr_schedule --lr_decay_steps 5

# APCI-FGSM Attack
#$BASE_CMD --attack apci_fgsm --eps 0.3 --steps 10 --beta1 0.9 --beta2 0.999 --learning_rate 0.01 --weight_decay 0.004 --epsilon 1e-7 --global_clipnorm 1.0

# API-FGSM Attack
#$BASE_CMD --attack api_fgsm --eps 0.3 --steps 10 --beta1 0.9 --beta2 0.9995 --learning_rate 0.0005 --epsilon 1e-8 --weight_decay 0.01 --delta 0.1 --wd_ratio 0.1 --nesterov (46.6%)

# SGDP-FGSM Attack
#$BASE_CMD --attack sgdp_fgsm --eps 0.3 --steps 10 --decay 1.0 --momentum 0.95 --dampening 0.1

# NAI-FGSM Attack
#$BASE_CMD --attack nai_fgsm --eps 0.3 --steps 10 --beta1 0.9 --beta2 0.999 --learning_rate 0.001 --epsilon 1e-7 --decay 0.96

# SOAP-FGSM Attack
#$BASE_CMD --attack soap_fgsm --eps 0.3 --steps 10 --beta1 0.95 --beta2 0.95 --learning_rate 0.001 --epsilon 1e-8 --preconditioning_freq 10

# AMI-FGSM Attack
$BASE_CMD --attack pmai_fgsm --eps 0.3 --steps 10 --decay 1.0 --m 5
#$BASE_CMD --attack tmi_fgsm --eps 0.3 --steps 10 --decay 1.0 --m 5
