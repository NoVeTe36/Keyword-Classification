import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import argparse
import numpy as np
import json
from pathlib import Path

# Now these imports should work
from utils.misc import get_model
from utils.dataset import GoogleSpeechDataset
from config_parser import get_config

def main():
    # Set paths - update the path to use the "no" audio file
    model_path = "runs/kwt1_finetune_mean_self_train/best.pth"
    audio_path = "speech_commands_v0.02/nine/1816b768_nohash_2.wav"  # Changed to the "no" sample
    config_path = "KWT_configs/kwt1_finetune_mean_config.yaml"
    
    # Parse config
    config = get_config(config_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load label mapping
    with open(config["label_map"], "r") as f:
        label_map_dict = json.load(f)  # Load the actual dictionary
        # Convert string keys to integers and reverse the mapping
        idx_2_label = {int(k): v for k, v in label_map_dict.items()}
    
    # Create model
    model = get_model(config["hparams"]["model"])
    
    # Load model weights
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Print information about the file to be processed
    print(f"Processing audio file: {audio_path}")
    
    # Create a simple dataset and get a sample
    dataset = GoogleSpeechDataset(
        data_list=[audio_path],  # First parameter should be data_list
        audio_settings=config["hparams"]["audio"],  # Should be the audio settings, not the full config
        label_map=label_map_dict  # Pass the dictionary, not the path
    )
    feature, _ = dataset[0]
    
    # Convert to tensor and add batch dimension
    audio_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(audio_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top prediction
        _, predicted_idx = torch.max(outputs, 1)
        predicted_idx = predicted_idx.item()
        predicted_label = idx_2_label.get(predicted_idx, "Unknown")
        confidence = probabilities[0][predicted_idx].item() * 100
        
        print(f"\nPrediction: {predicted_label} (class {predicted_idx})")
        print(f"Confidence: {confidence:.2f}%")
        
        # Display top 3 predictions
        print("\nTop 3 predictions:")
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        for i in range(3):
            idx = top3_idx[0][i].item()
            prob = top3_prob[0][i].item() * 100
            label = idx_2_label.get(idx, "Unknown")
            print(f"{label} (class {idx}): {prob:.2f}%")

if __name__ == "__main__":
    main()
