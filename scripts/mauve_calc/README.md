```bash
runai submit \
  --name mauve-calc-128-2 \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 0 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/mauve_calc/128-2.sh
    "
```


```bash
runai submit \
  --name mauve-calc-256 \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 0 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/mauve_calc/256.sh
    "
```


```bash
runai submit \
  --name mauve-calc-512 \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 0 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/mauve_calc/512.sh
    "
```


```bash
runai submit \
  --name mauve-calc-32 \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 0 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/mauve_calc/32.sh
    "
```



```bash
runai submit \
  --name mauve-calc-64 \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 0 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/mauve_calc/64.sh
    "
```

```bash
runai submit \
  --name mauve-calc-1024 \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 0 \
  --cpu 16 \
  --backoff-limit 0 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/mauve_calc/1024.sh
    "
```