

# Our-MDLM


```bash
runai submit \
  --name our-mdlm \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/mdlm_sas.sh --revise_step false --remove_self_attn false --mask_embedding_blending false
    "
```



# Our-MDLM + Conf-Remasking

```bash
runai submit \
  --name our-mdlm-conf-remasking \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/mdlm_sas.sh --revise_step true --remove_self_attn false --mask_embedding_blending false
    "
```


#  Our-MDLM + Conf-Remasking + Self-Att-Removal

```bash
runai submit \
  --name our-mdlm-conf-remasking-self-att-removal \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/mdlm_sas.sh --revise_step true --remove_self_attn true --mask_embedding_blending false
    "
```


# Our-MDLM + Conf-Remasking + Mask-Emb-Blending

```bash
runai submit \
  --name our-mdlm-conf-remasking-mask-emb-blending \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/mdlm_sas.sh --revise_step true --remove_self_attn false --mask_embedding_blending true
    "
```


# Our-MDLM + Conf-Remasking + Mask-Emb-Blending + Self-Att-Removal

```bash
runai submit \
  --name our-mdlm-conf-remasking-full-shortcut-rm \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/mdlm_sas.sh --revise_step true --remove_self_attn true --mask_embedding_blending true
    "
```




# Our-SEDD  ==> Done

 

```bash
runai submit \
  --name our-sedd \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/sedd.sh --revise_step false --remove_self_attn false --mask_embedding_blending false
    "
```



# Our-sedd + Conf-Remasking ==> Done

```bash
runai submit \
  --name our-sedd-conf-remasking \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/sedd.sh --revise_step true --remove_self_attn false --mask_embedding_blending false
    "
```


#  Our-sedd + Conf-Remasking + Self-Att-Removal ==> Done

```bash
runai submit \
  --name our-sedd-conf-remasking-self-att-removal \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/sedd.sh --revise_step true --remove_self_attn true --mask_embedding_blending false
    "
```


# Our-MDLM + Conf-Remasking + Mask-Emb-Blending ==> Done

```bash
runai submit \
  --name our-sedd-conf-remasking-mask-emb-blending \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/sedd.sh --revise_step true --remove_self_attn false --mask_embedding_blending true
    "
```


# Our-sedd + Conf-Remasking + Mask-Emb-Blending + Self-Att-Removal ==> Done

```bash
runai submit \
  --name our-sedd-conf-remasking-full-shortcut-rm \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/our-mdlm/sedd.sh --revise_step true --remove_self_attn true --mask_embedding_blending true
    "
```



# Temperal Command:

```bash
runai submit \
  --name ablation-mdlm \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_mdlm_sas.sh
    "
```
```bash
runai submit \
  --name ablation-sedd \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_sedd.sh
    "
```
```bash
runai submit \
  --name ablation-sedd-remdm \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_sedd_remdm.sh
    "
```
```bash
runai submit \
  --name ablation-sedd-original \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_sedd_original.sh
    "
```
```bash
runai submit \
  --name ablation-sedd-cap \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_sedd_remdm_cap.sh
    "
```

```bash
runai submit \
  --name sas-component-ablation \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/sas_component_ablation.sh
    "
```


===

```bash
runai submit \
  --name ablation-mdlm-fb \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_mdlm_fb.sh
    "
```
```bash
runai submit \
  --name ablation-mdlm-distill-sas \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_mdlm_distill_sas.sh
    "
```
```bash
runai submit \
  --name ablation-mdlm-distill-rescale \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_mdlm_distill_rescale.sh
    "
```
```bash
runai submit \
  --name ablation-mdlm-distill-cap \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_mdlm_distill_cap.sh
    "
```
```bash
runai submit \
  --name ablation-mdlm-distill-ancestral \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/ablation_mdlm_distill_ancestral.sh
    "
```
=====


```bash
runai submit \
  --name mauve-ablation-mdlm-sas \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/MAUVE_ablation_mdlm_sas.sh
    "
```


```bash
runai submit \
  --name mauve-ablation-mdlm-remdm-rescale \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/MAUVE_ablation_mdlm_remdm_rescale.sh
    "
```



```bash
runai submit \
  --name mauve-ablation-mdlm-remdm-loop \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/MAUVE_ablation_mdlm_remdm_loop.sh
    "
```



```bash
runai submit \
  --name mauve-ablation-mdlm-cap \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/shortcut-aware-sampling/MAUVE_ablation_mdlm_remdm_cap.sh
    "
```