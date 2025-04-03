"""Train the final model using the combined dataset with pseudo-labels"""
import os
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import wandb

from utils.dataset import GoogleSpeechDataset, get_loader
from utils.misc import get_model, seed_everything
from utils.trainer import train, evaluate
from utils.opt import get_optimizer
from utils.scheduler import get_scheduler, WarmUpLR
from utils.loss import LabelSmoothingLoss
from config_parser import get_config
from pathlib import Path

def validate_file_paths(file_list_path, base_dir=None):
    """Validate and clean a list of file paths."""
    with open(file_list_path, "r") as f:
        lines = f.read().strip().split("\n")
    
    valid_lines = []
    invalid_count = 0
    
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            invalid_count += 1
            continue
            
        # For pseudo-labeled data (path\tlabel)
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"Warning: Invalid format at line {i+1}: '{line}'")
                invalid_count += 1
                continue
                
            path = parts[0].strip()
            label = parts[1].strip()
            
            # Validate path
            if not path or len(path) < 5 or not label.isdigit():
                print(f"Warning: Invalid path or label at line {i+1}: '{line}'")
                invalid_count += 1
                continue
            
            # Check if the file exists as-is (without base_dir)
            if os.path.exists(path) and os.path.isfile(path):
                valid_lines.append(f"{path}\t{label}")
                continue
                
            # Try with base_dir if needed
            if not os.path.isabs(path) and base_dir:
                # Handle case where path already has part of base_dir
                if path.startswith("speech_commands_v0.02/"):
                    # Just use the path as is, don't prepend base_dir
                    full_path = path
                else:
                    full_path = os.path.join(base_dir, path)
                
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    valid_lines.append(f"{full_path}\t{label}")
                else:
                    print(f"Warning: File not found: '{full_path}'")
                    invalid_count += 1
            else:
                print(f"Warning: File not found: '{path}'")
                invalid_count += 1
        else:
            # For regular file paths
            path = line.strip()
            if not path or len(path) < 5:
                print(f"Warning: Invalid path at line {i+1}: '{path}'")
                invalid_count += 1
                continue
            
            # Check if the file exists as-is (without base_dir)
            if os.path.exists(path) and os.path.isfile(path):
                valid_lines.append(path)
                continue
                
            # Try with base_dir if needed
            if not os.path.isabs(path) and base_dir:
                # Handle case where path already has part of base_dir
                if path.startswith("speech_commands_v0.02/"):
                    # Just use the path as is, don't prepend base_dir
                    full_path = path
                else:
                    full_path = os.path.join(base_dir, path)
                
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    valid_lines.append(full_path)
                else:
                    print(f"Warning: File not found: '{full_path}'")
                    invalid_count += 1
            else:
                print(f"Warning: File not found: '{path}'")
                invalid_count += 1
    
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid entries out of {len(lines)} total entries")
    
    # Debug information
    if valid_lines:
        print("Sample valid lines:")
        for i, line in enumerate(valid_lines[:3]):
            print(f"  {i+1}: '{line}'")
    else:
        print("WARNING: No valid lines found!")
        
    return valid_lines

class CombinedDataset(Dataset):
    def __init__(self, data_file, config, train=True):
        self.config = config
        self.samples = []
        self.labels = []
        
        # Parse the combined dataset file
        with open(data_file, "r") as f:
            lines = f.read().strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "\t" in line:  # Pseudo-labeled format: path\tlabel
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                    
                path, label = parts[0].strip(), parts[1].strip()
                # Skip invalid paths
                if not os.path.exists(path) or not os.path.isfile(path):
                    print(f"Skipping non-existent file: {path}")
                    continue
                    
                try:
                    label_idx = int(label)
                    self.samples.append(path)
                    self.labels.append(label_idx)
                except ValueError:
                    print(f"Invalid label format: {label}")
                    continue
            else:  # Original labeled format: path
                path = line.strip()
                # Skip invalid paths
                if not os.path.exists(path) or not os.path.isfile(path):
                    print(f"Skipping non-existent file: {path}")
                    continue
                    
                # Extract label from path format: /path/to/label/file.wav
                try:
                    label_name = path.split("/")[-2]
                    
                    # Get label index from config
                    with open(config["label_map"], "r") as lf:
                        label_map = json.load(lf)
                    label_2_idx = {v: int(k) for k, v in label_map.items()}
                    
                    if label_name not in label_2_idx:
                        print(f"Warning: Unknown label {label_name} for file {path}")
                        continue
                        
                    label_idx = label_2_idx[label_name]
                    self.samples.append(path)
                    self.labels.append(label_idx)
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
                    continue
        
        print(f"Loaded {len(self.samples)} valid samples for {'training' if train else 'validation'}")
        
        # Create the dataset with label_map=None to avoid auto-label extraction
        # Fix: Pass samples directly without label_list argument
        self.dataset = GoogleSpeechDataset(
            self.samples,
            audio_settings=config["hparams"]["audio"],
            aug_settings=config["hparams"]["augment"] if train else None,
            label_map=None,  # Pass None to prevent auto-extraction of labels
            cache=config["exp"]["cache"]
        )
        
        # Manually set the label list since we already extracted them
        self.dataset.label_list = self.labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def get_combined_loader(data_file, config, train=True):
    dataset = CombinedDataset(data_file, config, train)
    
    loader = DataLoader(
        dataset, 
        batch_size=config["hparams"]["batch_size"],
        shuffle=train, 
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"]
    )
    return loader

def main(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])
    
    # Handle base directory more carefully
    # If the path is already absolute, don't use a base directory
    if os.path.isabs(args.combined_data):
        base_dir = None
    else:
        # Get the directory containing the combined dataset file
        base_dir = os.path.dirname(os.path.abspath(args.combined_data))
        if not base_dir:  # Empty means it's in current directory
            base_dir = os.getcwd()
    
    print(f"Using base directory: {base_dir}")
    
    # Override batch size if specified
    if args.batch_size:
        config["hparams"]["batch_size"] = args.batch_size
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Update experiment name
    config["exp"]["exp_name"] = f"{config['exp']['exp_name']}_self_train"

    # Add wandb initialization code
    if config["exp"].get("wandb", False):
        if config["exp"].get("wandb_api_key") is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()
        elif os.environ.get("WANDB_API_KEY", False):
            print("Found API key from env variable.")
        else:
            wandb.login()
        
        # Initialize wandb
        wandb.init(
            project=config["exp"].get("proj_name", "data2vec-KWS-self-training"),
            name=config["exp"]["exp_name"],
            config=config["hparams"]
        )
        # Set to use wandb
        use_wandb = True
    else:
        # Set not to use wandb
        use_wandb = False

    # Set save_dir by combining exp_dir and exp_name
    if "save_dir" not in config["exp"]:
        config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    # Create the directory if it doesn't exist
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    
    # Clean and validate file lists
    print("Validating combined dataset...")
    combined_data_valid = validate_file_paths(args.combined_data, base_dir)
    
    # Write cleaned lists to temporary files
    tmp_combined_path = args.combined_data + ".clean"
    with open(tmp_combined_path, "w") as f:
        f.write("\n".join(combined_data_valid))
    
    print("Validating validation dataset...")
    val_list_valid = validate_file_paths(args.val_list, base_dir)
    tmp_val_path = args.val_list + ".clean"
    with open(tmp_val_path, "w") as f:
        f.write("\n".join(val_list_valid))
    
    # Create dataloaders using the cleaned lists
    print("Creating training dataloader...")
    trainloader = get_combined_loader(tmp_combined_path, config, train=True)
    
    print("Creating validation dataloader...")
    # Read the validation file content first
    with open(tmp_val_path, "r") as f:
        val_paths = f.read().strip().split("\n")
    # Now pass the list of paths to get_loader
    valloader = get_loader(val_paths, config, train=False)
    
    # Create model
    model = get_model(config["hparams"]["model"])
    
    # Load checkpoint if provided
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.ckpt}")
    
    model = model.to(device)
    
    # Training setup
    criterion = LabelSmoothingLoss(
        num_classes=config["hparams"]["model"]["num_classes"],
        smoothing=config["hparams"]["l_smooth"]
    )
    optimizer = get_optimizer(model, config["hparams"]["optimizer"])
    
    scheduler_type = config["hparams"]["scheduler"]["scheduler_type"]
    max_epochs = config["hparams"]["scheduler"]["max_epochs"]
    if scheduler_type == "cosine_annealing":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=0
        )
    elif scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=config["hparams"]["scheduler"].get("step_size", 30),
            gamma=config["hparams"]["scheduler"].get("gamma", 0.1)
        )
    else:
        print(f"Warning: Unknown scheduler type '{scheduler_type}', using no scheduler")
        scheduler = None
    
    warmup_scheduler = None
    if "n_warmup" in config["hparams"]["scheduler"] and config["hparams"]["scheduler"]["n_warmup"] > 0:
        # Original code with error:
        # warmup_scheduler = WarmUpLR(
        #     optimizer, 
        #     init_lr=config["hparams"]["scheduler"].get("warmup_init_lr", 1e-6),
        #     num_warmup=config["hparams"]["scheduler"]["n_warmup"]
        # )
        
        # Fixed code based on the actual WarmUpLR implementation:
        n_warmup = config["hparams"]["scheduler"]["n_warmup"]
        # In your implementation, total_iters is the warmup steps
        warmup_scheduler = WarmUpLR(
            optimizer,
            total_iters=n_warmup,  # This is the parameter it actually accepts
            last_epoch=-1  # Default value
        )
        
    schedulers = {
        "scheduler": scheduler,
        "warmup": warmup_scheduler
    }
    
    # Train the model
    train(
        net=model,
        trainloader=trainloader,
        valloader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        schedulers=schedulers,
        config=config
    )

    # After training, finish wandb run if it was used
    if config["exp"].get("wandb", False):
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="Path to config file")
    parser.add_argument("--combined_data", required=True, help="Path to combined dataset file")
    parser.add_argument("--val_list", required=True, help="Path to validation list file")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint file (optional)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    args = parser.parse_args()
    
    main(args)