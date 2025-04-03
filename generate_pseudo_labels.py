"""Generate pseudo-labels for unlabelled data using the finetuned model"""
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from utils.dataset import get_loader
from utils.misc import get_model
from config_parser import get_config

def main(args):
    # Load configuration
    config = get_config(args.conf)
    device = torch.device(config["hparams"]["device"])
    
    # Load pseudo-candidates
    with open(args.candidates_file, "r") as f:
        pseudo_candidates = f.read().rstrip().split("\n")
    
    print(f"Loaded {len(pseudo_candidates)} samples for pseudo-labeling")
    
    # Create dataloader for the candidates
    pseudo_loader = get_loader(pseudo_candidates, config, train=False)
    
    # Load the finetuned model
    model = get_model(config["hparams"]["model"])
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Verify GPU usage
    print(f"Is model on CUDA: {next(model.parameters()).is_cuda}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name()}")
    
    # Generate pseudo-labels
    all_paths = []
    all_predictions = []
    all_confidences = []
    
    # Debug the first batch
    print("Checking dataloader output format...")
    batch_sample = next(iter(pseudo_loader))
    print(f"Batch type: {type(batch_sample)}")
    if isinstance(batch_sample, list):
        print(f"First element type: {type(batch_sample[0])}")
        print(f"Batch length: {len(batch_sample)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(pseudo_loader, desc="Generating pseudo-labels")):
            # Handle different batch formats
            if isinstance(batch, torch.Tensor):
                inputs = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                # Assume batch[0] is data and batch[1] is label
                inputs = batch[0].to(device)
            else:
                print(f"Unexpected batch type: {type(batch)}")
                continue
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions and confidence
            probs = torch.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probs, dim=1)
            
            # Add batch predictions
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_confidences.extend(confidence.cpu().numpy().tolist())
            
            # Track which files these predictions correspond to
            batch_size = inputs.size(0)
            start_idx = batch_idx * config["hparams"]["batch_size"]
            for i in range(batch_size):
                idx = start_idx + i
                if idx < len(pseudo_candidates):
                    all_paths.append(pseudo_candidates[idx])
    
    # Make sure we have the right number of predictions
    if len(all_paths) != len(all_predictions):
        print(f"Warning: Number of paths ({len(all_paths)}) doesn't match number of predictions ({len(all_predictions)})")
        # Use the minimum length to ensure they match
        min_length = min(len(all_paths), len(all_predictions))
        all_paths = all_paths[:min_length]
        all_predictions = all_predictions[:min_length]
        all_confidences = all_confidences[:min_length]
    
    # Apply confidence threshold if specified
    if args.confidence_threshold > 0:
        confident_indices = np.where(np.array(all_confidences) >= args.confidence_threshold)[0]
        confident_samples = [all_paths[i] for i in confident_indices]
        confident_labels = [all_predictions[i] for i in confident_indices]
        confident_confs = [all_confidences[i] for i in confident_indices]
        
        print(f"Keeping {len(confident_indices)} samples out of {len(all_paths)} based on confidence threshold {args.confidence_threshold}")
    else:
        confident_samples = all_paths
        confident_labels = all_predictions
        confident_confs = all_confidences
    
    # Save pseudo-labels
    with open(args.output_file, "w") as f:
        for sample, label, conf in zip(confident_samples, confident_labels, confident_confs):
            f.write(f"{sample}\t{label}\t{conf:.6f}\n")
    
    print(f"Generated {len(confident_samples)} pseudo-labels, saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="Path to config file")
    parser.add_argument("--model_path", required=True, help="Path to finetuned model checkpoint")
    parser.add_argument("--candidates_file", required=True, help="Path to file with pseudo-candidates")
    parser.add_argument("--output_file", required=True, help="Path to save pseudo-labels")
    parser.add_argument("--confidence_threshold", type=float, default=0.9, help="Minimum confidence score to keep")
    args = parser.parse_args()
    
    main(args)