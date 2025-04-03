import os
import argparse
import random

def main(args):
    # Load original pretraining list (80% unlabeled)
    with open(args.pretrain_list, "r") as f:
        pretrain_list = f.read().rstrip().split("\n")
    
    total_samples = len(pretrain_list)
    print(f"Total unlabeled samples: {total_samples}")
    
    # Shuffle the list
    random.seed(args.seed)
    random.shuffle(pretrain_list)
    
    # Split into two halves
    first_half = pretrain_list[:total_samples//2]
    second_half = pretrain_list[total_samples//2:]
    
    # Save the splits
    with open(args.output_dir + "/pretraining_first_half.txt", "w") as f:
        f.write("\n".join(first_half))
    
    with open(args.output_dir + "/pretraining_second_half.txt", "w") as f:
        f.write("\n".join(second_half))
    
    print(f"Saved {len(first_half)} samples for pretraining")
    print(f"Saved {len(second_half)} samples for pseudo-labeling")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_list", required=True, help="Path to original pretraining list")
    parser.add_argument("--output_dir", required=True, help="Directory to save the splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()
    
    main(args)