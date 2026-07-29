'''
python -u -m random_perturbation_experiment \
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
    sampling.num_sample_batches=40 \
    sampling.generated_seqs_path="${PWD}/outputs/mdlm-sas-32.json" \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remasking-via-shortcut-removal" \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false

'''
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

import json
import mauve

import torch
from pathlib import Path
import noise_schedule

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)



@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)



@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)
    
    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    logger.info('Generating samples.')
    model = _load_from_checkpoint(config=config,
                                  tokenizer=tokenizer)
    model.gen_ppl_metric.reset()
    if config.eval.disable_ema:
      logger.info('Disabling EMA.')
      model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    model = model.to("cuda")


    text = Path("/home/nafez/scratch/dllm-revision-sampler/text_clean.txt").read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    chunk_size = 1024
    vocab_size = tokenizer.vocab_size
    n_chunks = len(token_ids) // chunk_size
    print(f"Loaded {len(token_ids)} tokens ({n_chunks} chunks of {chunk_size})")
    # mask_id = tokenizer.mask_token_id
    mask_id = 50257
    eps = 1e-5
    steps = 1024
    i = 1023
    
    def _process_sigma(sigma):
        if sigma is None:
          assert self.parameterization == 'ar'
          return sigma
        if sigma.ndim > 1:
          sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
          sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma
    # timesteps = model._get_sampling_time_profile(eps, steps)
    noise = noise_schedule.get_noise(config,
                                      dtype=torch.float64)
    t = torch.tensor(0.99, device=model.device)   # or device=model.device
    sigma_t, _ = noise(t)

    # Tracking Variables (Pass-1, Pass-5, and Pass-10 counters)
    total_correct = 0
    total_correct_top5 = 0
    total_correct_top10 = 0
    
    total_copy_correct = 0
    total_copy_correct_top5 = 0
    total_copy_correct_top10 = 0
    
    total_corrupted = 0
    total_clean = 0
    
    total_clean_token_correct = 0
    total_clean_token_correct_top5 = 0
    total_clean_token_correct_top10 = 0
    
    model.eval()
    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            clean_tokens = torch.tensor(
                token_ids[start:end],
                dtype=torch.long,
                device=model.device,
            ).unsqueeze(0)  # [1, 1024]
            
            x = clean_tokens.clone()
            corruption_mask = (torch.rand_like(x.float()) < 0.20)
            random_tokens = torch.randint(low=0, high=vocab_size, size=x.shape, device=x.device,)
            x[corruption_mask] = random_tokens[corruption_mask]
            # x[corruption_mask] = mask_id
            
            num_corrupted = corruption_mask.sum().item()
            num_clean = (~corruption_mask).sum().item()
            
            if num_corrupted == 0:
                continue
                
            # t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
            # _, alpha_t = model.noise(t)
            # sigma = model._sigma_from_alphat(alpha_t)
            # sigma = model._process_sigma(sigma)
            sigma_t = sigma_t.expand(x.size(0))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                # logits = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=False, remove_self_attn=False)#.argmax(dim=-1)
                # logits = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=True, remove_self_attn=False)#.argmax(dim=-1)
                # logits = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=False, remove_self_attn=True)#.argmax(dim=-1)
                # logits = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=True, remove_self_attn=True)# .argmax(dim=-1)
                logits = model(x, sigma_t, mask_embedding_blending=False)

            
            # logits = torch.where(~corruption_mask.unsqueeze(-1), logits, torch.tensor(-float('inf'), device=logits.device))
            # --- PASS-1 PREDICTIONS ---
            pred_tokens = logits.argmax(dim=-1)
            
            # --- TOP-K PREDICTIONS (K=10 covers both top-5 and top-10) ---
            top10_preds = logits.topk(k=10, dim=-1).indices # Shape: [1, 1024, 10]
            
            # Broadcast masks for Top-5 and Top-10 evaluations
            # Slicing top10_preds[:, :, :5] gives the top 5 indices efficiently
            correct_top5_mask = (top10_preds[:, :, :5] == clean_tokens.unsqueeze(-1)).any(dim=-1)
            correct_top10_mask = (top10_preds == clean_tokens.unsqueeze(-1)).any(dim=-1)
            
            copy_top5_mask = (top10_preds[:, :, :5] == x.unsqueeze(-1)).any(dim=-1)
            copy_top10_mask = (top10_preds == x.unsqueeze(-1)).any(dim=-1)
            
            # --- TALLYING PASS-1 ---
            correct = ((pred_tokens == clean_tokens) & corruption_mask).sum().item()
            copy_correct = ((pred_tokens == x) & corruption_mask).sum().item()
            clean_token_correct = ((pred_tokens == clean_tokens) & (~corruption_mask)).sum().item()
            
            # --- TALLYING PASS-5 ---
            correct_top5 = (correct_top5_mask & corruption_mask).sum().item()
            copy_correct_top5 = (copy_top5_mask & corruption_mask).sum().item()
            clean_token_correct_top5 = (correct_top5_mask & (~corruption_mask)).sum().item()
            
            # --- TALLYING PASS-10 ---
            correct_top10 = (correct_top10_mask & corruption_mask).sum().item()
            copy_correct_top10 = (copy_top10_mask & corruption_mask).sum().item()
            clean_token_correct_top10 = (correct_top10_mask & (~corruption_mask)).sum().item()
            
            # --- ACCUMULATION ---
            total_correct += correct
            total_correct_top5 += correct_top5
            total_correct_top10 += correct_top10
            
            total_copy_correct += copy_correct
            total_copy_correct_top5 += copy_correct_top5
            total_copy_correct_top10 += copy_correct_top10
            
            total_corrupted += num_corrupted
            total_clean += num_clean
            
            total_clean_token_correct += clean_token_correct
            total_clean_token_correct_top5 += clean_token_correct_top5
            total_clean_token_correct_top10 += clean_token_correct_top10
            
    # Calculate Final Percentages
    model_acc = 100 * total_correct / max(total_corrupted, 1)
    model_acc_top5 = 100 * total_correct_top5 / max(total_corrupted, 1)
    model_acc_top10 = 100 * total_correct_top10 / max(total_corrupted, 1)
    
    copy_acc = 100 * total_copy_correct / max(total_corrupted, 1)
    copy_acc_top5 = 100 * total_copy_correct_top5 / max(total_corrupted, 1)
    copy_acc_top10 = 100 * total_copy_correct_top10 / max(total_corrupted, 1)
    
    clean_acc = 100 * total_clean_token_correct / max(total_clean, 1)
    clean_acc_top5 = 100 * total_clean_token_correct_top5 / max(total_clean, 1)
    clean_acc_top10 = 100 * total_clean_token_correct_top10 / max(total_clean, 1)
    
    overall_acc = 100 * (total_clean_token_correct + total_correct) / (total_corrupted + total_clean)
    overall_acc_top5 = 100 * (total_clean_token_correct_top5 + total_correct_top5) / (total_corrupted + total_clean)
    overall_acc_top10 = 100 * (total_clean_token_correct_top10 + total_correct_top10) / (total_corrupted + total_clean)
    
    print("===============Final Results===============")
    print(
        f"Denoising Accuracy (Corrupted Tokens):\n"
        f"  Pass-1:  {model_acc:.6f}% ({total_correct}/{total_corrupted})\n"
        f"  Pass-5:  {model_acc_top5:.6f}% ({total_correct_top5}/{total_corrupted})\n"
        f"  Pass-10: {model_acc_top10:.6f}% ({total_correct_top10}/{total_corrupted})"
    )
    print(
        f"Copying Rate (Corrupted Input -> Output):\n"
        f"  Pass-1:  {copy_acc:.6f}% ({total_copy_correct}/{total_corrupted})\n"
        f"  Pass-5:  {copy_acc_top5:.6f}% ({total_copy_correct_top5}/{total_corrupted})\n"
        f"  Pass-10: {copy_acc_top10:.6f}% ({total_copy_correct_top10}/{total_corrupted})"
    )
    print(
        f"Clean Accuracy (Clean Tokens):\n"
        f"  Pass-1:  {clean_acc:.6f}% ({total_clean_token_correct}/{total_clean})\n"
        f"  Pass-5:  {clean_acc_top5:.6f}% ({total_clean_token_correct_top5}/{total_clean})\n"
        f"  Pass-10: {clean_acc_top10:.6f}% ({total_clean_token_correct_top10}/{total_clean})"
    )
    print(
        f"Overall Accuracy (All Tokens):\n"
        f"  Pass-1:  {overall_acc:.6f}% ({(total_clean_token_correct + total_correct)}/{(total_corrupted + total_clean)})\n"
        f"  Pass-5:  {overall_acc_top5:.6f}% ({(total_clean_token_correct_top5 + total_correct_top5)}/{(total_corrupted + total_clean)})\n"
        f"  Pass-10: {overall_acc_top10:.6f}% ({(total_clean_token_correct_top10 + total_correct_top10)}/{(total_corrupted + total_clean)})"
    )




 

if __name__ == '__main__':
  main()