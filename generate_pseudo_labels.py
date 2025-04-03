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
    
    # Create dataloader for the candidates
    pseudo_loader = get_loader(pseudo_candidates, config, train=False)
    
    # Load the finetuned model
    model = get_model(config["hparams"]["model"])
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Generate pseudo-labels
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in tqdm(pseudo_loader, desc="Generating pseudo-labels"):
            inputs = batch.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_confidences.extend(confidence.cpu().numpy().tolist())
    
    # Apply confidence threshold if specified
    if args.confidence_threshold > 0:
        confident_indices = np.where(np.array(all_confidences) >= args.confidence_threshold)[0]
        confident_samples = [pseudo_candidates[i] for i in confident_indices]
        confident_labels = [all_predictions[i] for i in confident_indices]
        confident_confs = [all_confidences[i] for i in confident_indices]
        
        print(f"Keeping {len(confident_indices)} samples out of {len(pseudo_candidates)} based on confidence threshold {args.confidence_threshold}")
    else:
        confident_samples = pseudo_candidates
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
    
