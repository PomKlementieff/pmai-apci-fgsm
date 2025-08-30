# Periodic Momentum Amortization & Adaptive Perturbation Control (PMAI/APCI)
**TensorFlow/Keras implementation for iterative FGSM attacks**

> **TL;DR** — This repo implements two drop‑in upgrades for iterative FGSM:
> - **PMAI‑FGSM** (`pmai_fgsm`, `tmi_fgsm`): *Periodic Momentum Amortization* to damp Nesterov‑style oscillations and stabilize the sign direction.
> - **APCI‑FGSM** (`apci_fgsm`): *Adaptive Perturbation Control* with AdamW‑like per‑coordinate updates (in perturbation space), optional clipping, and decoupled decay.
>
> It also includes a broad set of baselines, input‑transform hybrids (DIM/TIM/SIM/PIM), and evaluation against common defenses (HGD, RS, R&P, Bit‑Red, FD, JPEG, NIPS‑r3, ComDefend, NRP).

---

## Repository layout

```
Adversarial_Attacks/
├── adversarial_methods.py          # Baselines: FGSM, I/MI/NI/PI-FGSM, EMI, SNI/BNI/DNI, QHMI, ANAGI, AI/API/NAI/SGDP/SOAP-FGSM...
├── ours.py                         # ✅ Ours: PMAI-FGSM, APCI-FGSM
├── hybrid_methods.py               # ✅ Hybrids for DIM/TIM/SIM/PIM
├── input_transformations.py        # DIM, TIM, SIM, PIM implementations
├── blackbox_attack_runner.py       # Transfer (substitute → target) evaluation
├── defense_models_attack.py        # Attacking defended targets
├── defense_attack_utils.py         # Loads defenses and handles their I/O/resize policies
├── attack_utils.py                 # Model/dataset utilities, Keras model zoo, ensemble loader
├── models/                         # Inception‑v4 module + robust ensemble checkpoints
├── third_party/                    # HGD, NIPS-r3, Bit-Red, ComDefend, FD, JPEG, NRP, R&P, RS
└── script/                         # Experiment scripts (hybrids/defenses; CSV→Excel formatting)
```

---

## Requirements

- Python ≥ 3.9
- TensorFlow ≥ 2.10 (tested on TF 2.x), Keras bundled
- NumPy, SciPy, scikit‑image, OpenCV (`opencv-python`), Pillow, h5py, matplotlib, openpyxl (for Excel export)
- CUDA/cuDNN+GPU recommended (mixed precision is enabled by default in the runners)

**Install**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy scipy scikit-image opencv-python pillow h5py matplotlib openpyxl
```

> **Checkpoints expected by the code**
>
> **Robust ensembles (for substitute/targets):**
> - `models/ens3_adv_inception_v3.ckpt-1`
> - `models/ens4_adv_inception_v3.ckpt-1`
> - `models/ens_adv_inception_resnet_v2.ckpt-1`
>
> **Defenses expecting weights under `third_party/` (see `defense_attack_utils.py`):**
> - JPEG / FD / Bit‑Red: `third_party/<JPEG|FD|Bit_Red>/ens3_adv_inception_v3.ckpt-1`
> - R&P (Inception‑ResNet‑v2): `third_party/R_P/ens_adv_inception_resnet_v2.ckpt`
> - RS (Random Smoothing): `third_party/RS/tf_model_weights.h5`
> - NRP (Purify): SavedModel at `third_party/NRP/tf_models/nrp_model`

---

## Dataset

- Point `--data_dir` to a folder containing ImageNet‑val `.JPEG` images (class subfolders are **not** required).
- The loaders take the first **1000** images by default. Resizing and preprocessing are handled per model.

> For a smoke test, 32–64 random `.JPEG` files are enough.

---

## Our methods (paper → code)

| Paper name | 1‑line intuition | Function |
|---|---|---|
| **PMAI‑FGSM** | Periodically amortize momentum to correct long‑horizon drift and stabilize the sign direction. | `pmai_fgsm(model, x, y, eps, steps, decay, m)`, `tmi_fgsm(...)` |
| **APCI‑FGSM** | AdamW‑like per‑coordinate update in **perturbation space** with optional gradient/parameter clipping. | `apci_fgsm(model, x, y, eps, steps, beta1, beta2, learning_rate, weight_decay, epsilon, amsgrad=False, clipnorm=None, clipvalue=None, global_clipnorm=None)` |

**Recommended defaults**: `eps=16/255`, `steps=10`, `alpha=eps/steps`. For quick demos, `eps=0.3` is used in scripts; for ImageNet‑style evals, prefer `8/255` or `16/255`.

---

## Quickstart (Python API)

```python
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from attack_utils import load_dataset, get_input_size, resize_batch
from ours import pmai_fgsm, apci_fgsm

sub = InceptionV3(weights='imagenet')
tgt = InceptionV3(weights='imagenet')
H, W = get_input_size('Inc-v3')

ds = load_dataset('/path/to/ILSVRC2012_img_val_subset', batch_size=8, image_size=(H, W))
for x in ds.take(1):
    y = tf.one_hot(tf.argmax(sub.predict(x), axis=1), depth=1000)

    # PMAI‑FGSM (m=5 amortization)
    x_adv_pmai = pmai_fgsm(sub, x, y, eps=16/255, steps=10, decay=1.0, m=5)

    # APCI‑FGSM (AdamW‑like update in perturbation space)
    x_adv_apci = apci_fgsm(sub, x, y, eps=16/255, steps=10,
                          beta1=0.9, beta2=0.999, learning_rate=0.01,
                          weight_decay=0.004, epsilon=1e-7, global_clipnorm=1.0)

    # Transfer to target
    y_nat = tf.argmax(tgt.predict(x), axis=1)
    y_adv = tf.argmax(tgt.predict(resize_batch(x_adv_pmai, (H, W))), axis=1)
    print('Transfer fooling rate:', float(tf.reduce_mean(tf.cast(y_adv != y_nat, tf.float32))))
```

---

## CLI — black‑box transfer

Main entry: **`blackbox_attack_runner.py`**

```bash
# Substitute: Inc-v3, Target: Res-101 (example)
python3 blackbox_attack_runner.py   --data_dir ./dataset --batch_size 32   --substitute_model Inc-v3 --target_model Res-101   --attack pmai_fgsm --eps 16/255 --steps 10 --decay 1.0 --m 5
```

**Available attacks (selection)**  
`fgsm, i_fgsm, mi_fgsm, ni_fgsm, pi_fgsm, emi_fgsm, sni_fgsm, bni_fgsm, dni_fgsm, qhmi_fgsm, anagi_fgsm, ai_fgsm, api_fgsm, nai_fgsm, sgdp_fgsm, soap_fgsm, pmai_fgsm, tmi_fgsm, apci_fgsm, hybrid_fgsm, hybrid_pmai_fgsm, hybrid_apci_fgsm`

### Hybrids (DIM/TIM/SIM/PIM)

```bash
# DIM + TIM (commonly helps)
python3 blackbox_attack_runner.py   --data_dir ./dataset --batch_size 32   --substitute_model Inc-v3 --target_model Res-101   --attack hybrid_fgsm --eps 16/255   --use_dim --use_tim   --resize_rate 1.1 --diversity_prob 0.5   --kernel_size 5
```

> **Tip**: DIM+TIM is a safe starting point. DIM+TIM+SIM may hurt on some setups — add SIM selectively.

Convenience script: `script/run_blackbox_attacks.sh` (examples for many attacks).

---

## Evaluating against defenses

Use **`defense_models_attack.py`** to attack defended targets (HGD, RS, R&P, Bit‑Red, Feature Distillation, JPEG, NIPS‑r3, ComDefend, NRP):

```bash
python3 defense_models_attack.py   --data_dir ./dataset --batch_size 32   --substitute_model Inc-v3   --target_model hgd   --attack pmai_fgsm --eps 16/255 --steps 10 --decay 1.0 --m 5
```

Hybrid variants are available too:

```bash
python3 defense_models_attack.py   --data_dir ./dataset --batch_size 32   --substitute_model Inc-v3   --target_model hgd   --attack hybrid_apci_fgsm --eps 16/255 --steps 10   --beta1 0.9 --beta2 0.999 --learning_rate 0.01 --weight_decay 0.004   --epsilon 1e-7 --global_clipnorm 1.0   --use_dim --use_tim --use_sim   --resize_rate 1.1 --diversity_prob 0.5 --kernel_size 5   --scale_factors 1.0 0.9 0.8 0.7 0.6
```

---

## Reproduce hybrids & defenses

- **single vs hybrid vs ours**  
  - Script: `script/run_hybrid_attacks.sh` for quick runs.  
  - Full sweep + CSV→Excel: `script/hybrid_attack_experiment.sh` →
    - `hybrid_attack_results.csv`  
    - `hybrid_attack_results.xlsx` (formatted; includes per‑target averages)

- **defenses**  
  - Script: `script/run_defense_attacks.sh` for quick checks.  
  - Full sweep + CSV→Excel: `script/defense_model_attack_experiment.sh` →
    - `defense_attack_results.csv`  
    - `defense_attack_results.xlsx` (formatted; includes overall averages)

---

## Model zoo (string → Keras model)

Use `--substitute_model` / `--target_model` with these codes (see `attack_utils.py`):

- **ConvNeXt**: `Con-B`
- **DenseNet**: `Den-121`, `Den-169`, `Den-201`
- **EffNetV2**: `Eff-v2_b0`
- **Inception**: `Inc-v3`, `Inc-v4`, **robust ensembles**: `Inc-v3_ens3`, `Inc-v3_ens4`
- **Inception‑ResNet**: `IncRes-v2`, **robust**: `IncRes-v2_ens`
- **MobileNetV3**: `Mob-v3_L`, `Mob-v3_S`
- **NASNet**: `Nas-M`
- **ResNet‑V2**: `Res-50`, `Res-101`, `Res-152`
- **Xception**: `Xce`

> Robust ensembles load from the checkpoint files listed in **Requirements → Checkpoints**.

---

## Defaults & recommendations

- **Budget**: `eps = 16/255`, `steps = 10`, `alpha = eps/steps`
- **PMAI**: `decay=1.0`, amortization length `m=5`
- **APCI**: `beta1=0.9`, `beta2=0.999`, `learning_rate=0.01`, `weight_decay=0.004`, `epsilon=1e-7`, optional `global_clipnorm=1.0`
- **Hybrids**: start with **DIM+TIM**; add **SIM** only if it helps on your setup
- **Normalization**: gradients are L1‑mean normalized by default (see comments in `ours.py`)

---

## Notes & troubleshooting

- **Mixed precision** is enabled in the runners. If your GPU/driver is old, disable it by removing the `mixed_float16` policy lines in the scripts.
- **Checkpoint errors** → verify the exact file names/paths shown above.
- **Third‑party defenses** → ensure the required assets exist under `third_party/<defense>/...` (NRP expects a SavedModel).
- **Dataset shape** → we auto‑resize per model; use `get_input_size(code)` to query sizes.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{pmai_apci_2025,
  title     = {Periodic Momentum Amortization and Adaptive Perturbation Control for Iterative FGSM Attacks on Vision Models},
  author    = {Authors omitted here},
  booktitle = {...},
  year      = {2025}
}
```
