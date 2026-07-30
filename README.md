# Overview

This repository provides the code used to evaluate **MDLM** and **SEDD** with our proposed **Shortcut-Aware Sampling (SAS)** strategy.

The implementation includes:

* Shortcut-Aware Sampling (SAS)
* Confidence-Based Remasking
* ReMDM baselines
* Evaluation scripts for MDLM and SEDD

# Getting Started
v
To get started, create a conda environment containing the required dependencies.

```bash
conda create -n duo python=3.12
conda activate duo
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
```

# Checkpoints

We use the publicly available checkpoints released by the authors of the [DUO Repository](https://github.com/s-sahoo/duo).

* **Baselines** (SEDD, MDLM): Trained on OpenWebText
  * [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing) — download `mdlm.ckpt`, `sedd.ckpt`.



# MDLM

## Shortcut-Aware Sampling (SAS)

Run MDLM with our proposed Shortcut-Aware Sampling strategy:

```bash
bash scripts/shortcut-aware-sampling/mdlm.sh \
    --revise_step true \
    --remove_self_attn true \
    --mask_embedding_blending true
```

## Confidence-Based Remasking

Run the standard confidence-remasking baseline:

```bash
bash scripts/shortcut-aware-sampling/mdlm.sh \
    --revise_step true \
    --remove_self_attn false \
    --mask_embedding_blending false
```

## ReMDM Baselines

### ReMDM-Rescale

```bash
bash scripts/remdm-rescale.sh
```

### ReMDM-Loop

```bash
bash scripts/remdm-loop.sh
```

# SEDD

## Shortcut-Aware Sampling (SAS)

```bash
bash scripts/shortcut-aware-sampling/sedd.sh \
    --revise_step true \
    --remove_self_attn true \
    --mask_embedding_blending true
```

## Original Sampler

```bash
bash scripts/sedd.sh
```

# EPFL RCP Setup and Commands

## Request an Interactive Session

```bash
runai submit \
  --name mdlm-sampler-interactive \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash /scratch/mnafez/remdm-shortcut-removal/epfl-rcp-bootstrap.sh
```

Connect to the session:

```bash
runai bash mdlm-sampler-interactive -- bash --login
```

### Bootstrap Script

See `epfl-rcp-bootstrap.sh` for:

* 8-hour interactive session configuration
* Automatic `/scratch` symbolic link creation
* Automatic Conda environment activation

# Running Shortcut-Aware Sampling in an Interactive Session

The following command launches MDLM evaluation using Shortcut-Aware Sampling:

```bash
python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path=/home/nafez/scratch/remdm-shortcut-removal/weights/mdlm.ckpt \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-shortcut-aware-sampling" \
    T=0 \
    sampling.steps=1024 \
    seed=1 \
    sampling.num_sample_batches=1 \
    sampling.generated_seqs_path=/home/nafez/scratch/remdm-shortcut-removal/outputs/remdm-conf_T-1024_topp-0.9.json \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=true \
    +model.remove_self_attn=true \
    +sampling.mask_embedding_blending=true
```


