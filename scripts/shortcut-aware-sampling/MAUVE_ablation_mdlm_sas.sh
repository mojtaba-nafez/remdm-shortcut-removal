#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export HYDRA_FULL_ERROR=1

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
    eval.checkpoint_path="${PWD}/weights/mdlm.ckpt" \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-shortcut-aware-sampling" \
    T=0 \
    sampling.steps=19 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/mdlm-sas-32.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=true \
    +model.remove_self_attn=true \
    +sampling.mask_embedding_blending=true



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
    eval.checkpoint_path="${PWD}/weights/mdlm.ckpt" \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-shortcut-aware-sampling" \
    T=0 \
    sampling.steps=37 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/mdlm-sas-64.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=true \
    +model.remove_self_attn=true \
    +sampling.mask_embedding_blending=true



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
    eval.checkpoint_path="${PWD}/weights/mdlm.ckpt" \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-shortcut-aware-sampling" \
    T=0 \
    sampling.steps=74 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/mdlm-sas-128.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=true \
    +model.remove_self_attn=true \
    +sampling.mask_embedding_blending=true



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
    eval.checkpoint_path="${PWD}/weights/mdlm.ckpt" \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-shortcut-aware-sampling" \
    T=0 \
    sampling.steps=147 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/mdlm-sas-256.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=true \
    +model.remove_self_attn=true \
    +sampling.mask_embedding_blending=true


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
    eval.checkpoint_path="${PWD}/weights/mdlm.ckpt" \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-shortcut-aware-sampling" \
    T=0 \
    sampling.steps=293 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/mdlm-sas-512.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=true \
    +model.remove_self_attn=true \
    +sampling.mask_embedding_blending=true


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
    eval.checkpoint_path="${PWD}/weights/mdlm.ckpt" \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-shortcut-aware-sampling" \
    T=0 \
    sampling.steps=586 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/mdlm-sas-1024.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=true \
    +model.remove_self_attn=true \
    +sampling.mask_embedding_blending=true


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
    eval.checkpoint_path="${PWD}/weights/mdlm.ckpt" \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-shortcut-aware-sampling" \
    T=0 \
    sampling.steps=1171 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/mdlm-sas-2048.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=true \
    +model.remove_self_attn=true \
    +sampling.mask_embedding_blending=true





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
    eval.checkpoint_path="${PWD}/weights/mdlm.ckpt" \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-shortcut-aware-sampling" \
    T=0 \
    sampling.steps=2341 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/mdlm-sas-4096.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=true \
    +model.remove_self_attn=true \
    +sampling.mask_embedding_blending=true

