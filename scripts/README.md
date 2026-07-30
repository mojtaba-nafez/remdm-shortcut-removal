

# Our-MDLM


```bash
bash scripts/our-mdlm/mdlm_sas.sh --revise_step false --remove_self_attn false --mask_embedding_blending false
```



# Our-MDLM + Conf-Remasking

```bash
bash scripts/our-mdlm/mdlm_sas.sh --revise_step true --remove_self_attn false --mask_embedding_blending false
```


#  Our-MDLM + Conf-Remasking + Self-Att-Removal

```bash
bash scripts/our-mdlm/mdlm_sas.sh --revise_step true --remove_self_attn true --mask_embedding_blending false
```


# Our-MDLM + Conf-Remasking + Mask-Emb-Blending

```bash
bash scripts/our-mdlm/mdlm_sas.sh --revise_step true --remove_self_attn false --mask_embedding_blending true
```


# Our-MDLM + Conf-Remasking + Mask-Emb-Blending + Self-Att-Removal

```bash
bash scripts/our-mdlm/mdlm_sas.sh --revise_step true --remove_self_attn true --mask_embedding_blending true
```




# Our-SEDD 

 

```bash
bash scripts/our-mdlm/sedd.sh --revise_step false --remove_self_attn false --mask_embedding_blending false
```



# Our-sedd + Conf-Remasking ==> Done

```bash
bash scripts/our-mdlm/sedd.sh --revise_step true --remove_self_attn false --mask_embedding_blending false
```


#  Our-sedd + Conf-Remasking + Self-Att-Removal ==> Done

```bash
bash scripts/our-mdlm/sedd.sh --revise_step true --remove_self_attn true --mask_embedding_blending false
```


# Our-MDLM + Conf-Remasking + Mask-Emb-Blending ==> Done

```bash
bash scripts/our-mdlm/sedd.sh --revise_step true --remove_self_attn false --mask_embedding_blending true
```


# Our-sedd + Conf-Remasking + Mask-Emb-Blending + Self-Att-Removal ==> Done

```bash
bash scripts/our-mdlm/sedd.sh --revise_step true --remove_self_attn true --mask_embedding_blending true
```
