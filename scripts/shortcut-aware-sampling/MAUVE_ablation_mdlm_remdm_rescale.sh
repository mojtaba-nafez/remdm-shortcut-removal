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
    hydra.run.dir="${PWD}/outputs/remdm-rescale" \
    T=0 \
    sampling.steps=32 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/rescale_remdm_mdlm_32.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remdm-rescale" \
    sampling.eta=0.015 \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false




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
    hydra.run.dir="${PWD}/outputs/remdm-rescale" \
    T=0 \
    sampling.steps=64 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/rescale_remdm_mdlm_64.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remdm-rescale" \
    sampling.eta=0.015 \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false



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
    hydra.run.dir="${PWD}/outputs/remdm-rescale" \
    T=0 \
    sampling.steps=128 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/rescale_remdm_mdlm_128.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remdm-rescale" \
    sampling.eta=0.015 \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false




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
    hydra.run.dir="${PWD}/outputs/remdm-rescale" \
    T=0 \
    sampling.steps=256 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/rescale_remdm_mdlm_256.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remdm-rescale" \
    sampling.eta=0.015 \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false



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
    hydra.run.dir="${PWD}/outputs/remdm-rescale" \
    T=0 \
    sampling.steps=512 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/rescale_remdm_mdlm_512.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remdm-rescale" \
    sampling.eta=0.015 \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false



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
    hydra.run.dir="${PWD}/outputs/remdm-rescale" \
    T=0 \
    sampling.steps=1024 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/rescale_remdm_mdlm_1024.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remdm-rescale" \
    sampling.eta=0.015 \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false


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
    hydra.run.dir="${PWD}/outputs/remdm-rescale" \
    T=0 \
    sampling.steps=2048 \
    seed=1 \
    sampling.num_sample_batches=625 \
    sampling.generated_seqs_path="${PWD}/outputs/rescale_remdm_mdlm_2048.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remdm-rescale" \
    sampling.eta=0.015 \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false

