"""Train the final model using the combined dataset with pseudo-labels"""
import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader

from utils.misc import get_model, seed_everything
from utils.trainer import train, evaluate
from utils.dataset import SpeechCommandsDataset
from utils.opt import get_optimizer
from utils.scheduler import get_scheduler, WarmUpLR
from utils.loss import LabelSmoothingLoss
from config_parser import get_config

class CombinedDataset(Dataset):
    def __init__(self, data_file, config, train=True):
        self.config = config
        self.samples = []
        self.labels = []
        
        # Parse the combined dataset file
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:  # Pseudo-labeled format: path\tlabel
                    path, label = line.split("\t")
                    self.samples.append(path)
                    self.labels.append(int(label))
                else:  # Original labeled format: path
                    self.samples.append(line)
                    label_name = line.split("/")[-2]  # Extract label from path
                    label_idx = config["label_map"][label_name]
                    self.labels.append(label_idx)
        
        # Create the dataset
        self.dataset = SpeechCommandsDataset(
            self.samples, 
            self.labels,
            sample_rate=config["hparams"]["sample_rate"],
            train=train
        )
    
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
        num_workers=config["hparams"]["num_workers"],
        pin_memory=True
    )
    return loader

def main(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])
    device = torch.device(config["hparams"]["device"])
    
    # Update experiment name
    if args.id is not None:
        config["exp"]["exp_name"] = f"{config['exp']['exp_name']}_{args.id}"
    else:
        config["exp"]["exp_name"] = f"{config['exp']['exp_name']}_with_pseudo"
    
    # Create dataloaders
    trainloader = get_combined_loader(args.combined_data, config, train=True)
    
    # Load validation data from original validation list
    with open(args.val_list, "r") as f:
        val_list = f.read().rstrip().split("\n")
    valloader = get_loader(val_list, config, train=False)
    
    # Create model
    model = get_model(config["hparams"]["model"])
    
    # Load checkpoint if provided
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    
    # Training setup
    criterion = LabelSmoothingLoss(
        classes=config["hparams"]["model"]["num_classes"],
        smoothing=config["hparams"]["label_smoothing"]
    )
    optimizer = get_optimizer(model, config["hparams"]["optimizer"])
    scheduler = get_scheduler(optimizer, config["hparams"]["scheduler"])
    
    warmup_scheduler = None
    if config["hparams"]["warmup_steps"] > 0:
        warmup_scheduler = WarmUpLR(
            optimizer, 
            init_lr=config["hparams"]["warmup_init_lr"],
            num_warmup=config["hparams"]["warmup_steps"]
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
    parser.add_argument("--id", default=None, help="Optional experiment identifier")
    args = parser.parse_args()
    
    main(args)