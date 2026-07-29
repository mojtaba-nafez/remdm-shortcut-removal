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


python compute_mauve.py "${PWD}/outputs//rescale_remdm_mdlm_128_mauve_inputs.json"


python compute_mauve.py "${PWD}/outputs//loop_remdm_mdlm_128_mauve_inputs.json"


python compute_mauve.py "${PWD}/outputs//cap_remdm_mdlm_128.json"


python compute_mauve.py "${PWD}/outputs//mdlm-sas-128_mauve_inputs.json"



