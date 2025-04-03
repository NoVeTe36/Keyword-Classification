"""Combine labeled data with pseudo-labeled data for the final training"""
import os
import argparse

def main(args):
    # Load original labeled data (just paths)
    with open(args.labeled_file, "r") as f:
        labeled_data = f.read().rstrip().split("\n")
    
    # Load pseudo-labeled data (path\tlabel\tconfidence)
    pseudo_data = []
    with open(args.pseudo_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                # Keep just the path and label
                sample, label = parts[0], parts[1]
                pseudo_data.append(f"{sample}\t{label}")
    
    # Create combined dataset
    combined_data = labeled_data + pseudo_data
    
    # Save combined dataset
    with open(args.output_file, "w") as f:
        f.write("\n".join(combined_data))
    
    print(f"Combined {len(labeled_data)} labeled samples with {len(pseudo_data)} pseudo-labeled samples")
    print(f"Total dataset size: {len(combined_data)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_file", required=True, help="Path to labeled data file")
    parser.add_argument("--pseudo_file", required=True, help="Path to pseudo-labeled data file")
    parser.add_argument("--output_file", required=True, help="Path to save combined dataset")
    args = parser.parse_args()
    
    main(args)