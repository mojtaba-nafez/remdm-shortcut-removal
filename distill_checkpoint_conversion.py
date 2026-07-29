import os
import torch
import pytorch_lightning as pl
from sdtt import load_scaling_student

def build_patched_checkpoint(size="sm", round_num=7, save_path="/home/nafez/scratch/remdm-shortcut-removal/weights/sdtt_student_sm_round7.ckpt"):
    print("--------------------------------------------------")
    print(f"Step 1: Loading SDTT student (size={size}, round={round_num})")
    print("--------------------------------------------------")
    # Load the official student model
    model = load_scaling_student(size=size, round=round_num)
    
    # Extract raw model state dict
    state_dict = model.state_dict()
    
    print("\n--------------------------------------------------")
    print("Step 2: Constructing EMA 'shadow_params'")
    print("--------------------------------------------------")
    # We query model.parameters() to guarantee the exact tensor shapes and order
    # that your codebase's self.ema.load_state_dict() expects.
    shadow_params = [p.detach().clone() for p in model.parameters()]
    print(f"Successfully cloned {len(shadow_params)} parameter tensors for EMA.")
    
    # Package EMA state exactly as expected by your ema.py:
    #   self.decay = state_dict['decay']
    #   self.num_updates = state_dict['num_updates']
    #   self.shadow_params = state_dict['shadow_params']
    ema_state = {
        "decay": 0.9999,
        "num_updates": 0,
        "shadow_params": shadow_params
    }
    
    print("\n--------------------------------------------------")
    print("Step 3: Mocking PyTorch Lightning loop states")
    print("--------------------------------------------------")
    # Constructing the exact nested dict matching the lookup:
    # checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
    loops_state = {
        "fit_loop": {
            "epoch_progress": {
                "current": {
                    "completed": 0
                }
            },
            "epoch_loop.batch_progress": {
                "current": {
                    "completed": 0
                }
            }
        }
    }
    
    print("\n--------------------------------------------------")
    print("Step 4: Compiling complete Lightning Checkpoint Dictionary")
    print("--------------------------------------------------")
    # Get active lightning version to prevent version incompatibility warnings
    pl_version = pl.__version__
    print(f"Using Lightning version metadata: {pl_version}")
    
    checkpoint_data = {
        "pytorch-lightning_version": pl_version,
        "state_dict": state_dict,
        "ema": ema_state,
        "loops": loops_state,
        "global_step": 0,
        "epoch": 0,
        "hyper_parameters": getattr(model, "hparams", {})
    }
    
    print("\n--------------------------------------------------")
    print(f"Step 5: Saving patched checkpoint to {save_path}")
    print("--------------------------------------------------")
    # Ensure destination folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the checkpoint
    torch.save(checkpoint_data, save_path)
    print("Success! Checkpoint successfully constructed and patched.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    build_patched_checkpoint(
        size="sm", 
        round_num=7, 
        save_path="/home/nafez/scratch/remdm-shortcut-removal/weights/sdtt_student_sm_round7.ckpt"
    )