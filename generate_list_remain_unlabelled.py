"""Script to identify samples for pseudo-labeling"""
import os
import argparse

def main(args):
    # Load original full list
    with open(os.path.join(args.data_dir, "training_list.txt"), "r") as f:
        original_train_full = set(f.read().rstrip().split("\n"))
    
    # Load the data splits we created
    with open(os.path.join(args.output_dir, "pretraining_list.txt"), "r") as f:
        pretraining_set = set(f.read().rstrip().split("\n"))
    
    with open(os.path.join(args.output_dir, "training_list.txt"), "r") as f:
        training_set = set(f.read().rstrip().split("\n"))
    
    # Find the remaining unlabelled data (not used for pretraining or training)
    pseudo_candidates = original_train_full - (pretraining_set | training_set)
    
    # Save the list of candidates for pseudo-labeling
    with open(os.path.join(args.output_dir, "pseudo_candidates.txt"), "w") as f:
        f.write("\n".join(pseudo_candidates))
    
    print(f"Found {len(pseudo_candidates)} candidates for pseudo-labeling")
    print(f"Pretraining set: {len(pretraining_set)} samples")
    print(f"Training set: {len(training_set)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", required=True, help="Path to original dataset directory")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to directory with generated lists")
    args = parser.parse_args()
    
    main(args)