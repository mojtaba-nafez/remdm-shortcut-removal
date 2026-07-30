# Breaking the Identity-Mapping Shortcut in Masked Diffusion Language Models

Official evaluation code for AAAI anonymous submission.

---

## Overview

This repository provides code for evaluating **MDLM** and **SEDD** using **Shortcut-Aware Sampling (SAS)**:
* Implementation of SAS for MDLM & SEDD
* Synthetic corruption experiments (Table 1)
* Standard baseline samplers

---

## Getting Started

```bash
conda create -n sas_eval python=3.12 -y
conda activate sas_eval
conda install nvidia/label/cuda-12.4.0::cuda-toolkit -c nvidia
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
```

Place model checkpoints in `./weights/` (`mdlm.ckpt`, `sedd.ckpt`). *(Links anonymized for double-blind review).*

---

## Reproducing Experiments

### Corruption Experiment (Table 1)
```bash
python -u -m random_perturbation_experiment     model=small backbone=dit model.length=1024     eval.checkpoint_path="${PWD}/weights/mdlm.ckpt"     +model.remove_self_attn=false     +sampling.mask_embedding_blending=false
```

### Shortcut-Aware Sampling (SAS)

**MDLM:**
```bash
bash scripts/mdlm_sas.sh --revise_step true --remove_self_attn true --mask_embedding_blending true
```

**SEDD:**
```bash
bash scripts/sedd_sas.sh --revise_step true --remove_self_attn true --mask_embedding_blending true
```

**Flags:**
* `--revise_step`: Confidence-driven revision step.
* `--remove_self_attn`: Diagonal Masked Self-Attention.
* `--mask_embedding_blending`: Mask Embedding Fusion.

---

## Acknowledgements

Built upon open-source implementations of **ReMDM** and **DUO** *(links redacted for anonymization)*.