#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

remove_self_attn=false
mask_embedding_blending=false
revise_step=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --remove_self_attn)
            remove_self_attn="$2"
            shift 2
            ;;
        --mask_embedding_blending)
            mask_embedding_blending="$2"
            shift 2
            ;;
        --revise_step)
            revise_step="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

checkpoint_path=weights/mdlm.ckpt

T=0
sampling_steps=1024
p=0.9
generated_seqs_path=outputs/our-mdlm_revise_step_${revise_step}_remove_self_attn_${remove_self_attn}_mask_embedding_blending_${mask_embedding_blending}_T-${sampling_steps}_topp-${p}.json


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
    eval.checkpoint_path=${checkpoint_path} \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/our-mdlm" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=200 \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.nucleus_p=${p} \
    sampling.sampler="remasking-via-shortcut-removal" \
    hydra.job.chdir=false \
    +model.remove_self_attn=${remove_self_attn} \
    +sampling.revise_step=${revise_step} \
    +sampling.mask_embedding_blending=${mask_embedding_blending}