"""Train the final model using the combined dataset with pseudo-labels"""
import os
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json

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
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            invalid_count += 1
            continue
            
        # For pseudo-labeled data (path\tlabel)
        if "\t" in line:
            parts = line.split("\t")
            path = parts[0].strip()
            if len(parts) < 2 or not path or len(path) < 3:
                invalid_count += 1
                continue
                
            # Check if path exists
            if base_dir and not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            
            if os.path.exists(path):
                valid_lines.append(line)
            else:
                print(f"Warning: File not found: {path}")
                invalid_count += 1
        else:
            # For regular file paths
            path = line.strip()
            if not path or len(path) < 3:  # Simple validation to catch obviously wrong paths
                invalid_count += 1
                continue
                
            # Check if path exists
            if base_dir and not os.path.isabs(path):
                path = os.path.join(base_dir, path)
                
            if os.path.exists(path):
                valid_lines.append(line)
            else:
                print(f"Warning: File not found: {path}")
                invalid_count += 1
    
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid entries out of {len(lines)} total entries")
        
    return valid_lines

class CombinedDataset(Dataset):
    def __init__(self, data_file, config, train=True):
        self.config = config
        self.samples = []
        self.labels = []
        
        # Parse the combined dataset file
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if "\t" in line:  # Pseudo-labeled format: path\tlabel
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                        
                    path, label = parts[0], parts[1]
                    # Validate the path
                    if not os.path.exists(path) or len(path) < 3:
                        continue
                        
                    self.samples.append(path)
                    self.labels.append(int(label))
                else:  # Original labeled format: path
                    # Validate the path
                    if not os.path.exists(line) or len(line) < 3:
                        continue
                        
                    self.samples.append(line)
                    # Extract label from path format: /path/to/label/file.wav
                    label_name = line.split("/")[-2]
                    
                    # Get label index from config
                    with open(config["label_map"], "r") as lf:
                        label_map = json.load(lf)
                    label_2_idx = {v: int(k) for k, v in label_map.items()}
                    
                    if label_name not in label_2_idx:
                        print(f"Warning: Unknown label {label_name} for file {line}")
                        continue
                        
                    label_idx = label_2_idx[label_name]
                    self.labels.append(label_idx)
        
        print(f"Loaded {len(self.samples)} valid samples for {'training' if train else 'validation'}")
        
        # Create the dataset
        self.dataset = GoogleSpeechDataset(
            data_list=self.samples, 
            label_map=None,  # We already have labels
            audio_settings=config["hparams"]["audio"],
            aug_settings=config["hparams"]["augment"] if train else None,
            cache=config["exp"]["cache"]
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]
        label = self.labels[idx]
        return x, label

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
    
    # Clean and validate file lists
    print("Validating combined dataset...")
    combined_data_valid = validate_file_paths(args.combined_data)
    
    # Write cleaned lists to temporary files
    tmp_combined_path = args.combined_data + ".clean"
    with open(tmp_combined_path, "w") as f:
        f.write("\n".join(combined_data_valid))
    
    print("Validating validation dataset...")
    val_list_valid = validate_file_paths(args.val_list)
    tmp_val_path = args.val_list + ".clean"
    with open(tmp_val_path, "w") as f:
        f.write("\n".join(val_list_valid))
    
    # Create dataloaders using the cleaned lists
    print("Creating training dataloader...")
    trainloader = get_combined_loader(tmp_combined_path, config, train=True)
    
    print("Creating validation dataloader...")
    valloader = get_loader(tmp_val_path, config, train=False)
    
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
        classes=config["hparams"]["model"]["num_classes"],
        smoothing=config["hparams"]["l_smooth"]
    )
    optimizer = get_optimizer(model, config["hparams"]["optimizer"])
    scheduler = get_scheduler(optimizer, config["hparams"]["scheduler"])
    
    warmup_scheduler = None
    if "n_warmup" in config["hparams"]["scheduler"] and config["hparams"]["scheduler"]["n_warmup"] > 0:
        warmup_scheduler = WarmUpLR(
            optimizer, 
            init_lr=config["hparams"]["scheduler"].get("warmup_init_lr", 1e-6),
            num_warmup=config["hparams"]["scheduler"]["n_warmup"]
        )
    
    # Train the model
    train(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        config=config,
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="Path to config file")
    parser.add_argument("--combined_data", required=True, help="Path to combined dataset file")
    parser.add_argument("--val_list", required=True, help="Path to validation list file")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint file (optional)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    args = parser.parse_args()
    
    main(args)