"""Script to identify samples for pseudo-labeling"""
import os
import argparse
import glob

def main(args):
    # Get all audio files in the dataset
    all_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith(".wav"):
                # Convert to relative path format used in the lists
                rel_path = os.path.join(os.path.basename(root), file)
                all_files.append(rel_path)
    
    # Load validation and test lists
    with open(os.path.join(args.data_dir, "validation_list.txt"), "r") as f:
        validation_list = set(f.read().rstrip().split("\n"))
    
    with open(os.path.join(args.data_dir, "testing_list.txt"), "r") as f:
        testing_list = set(f.read().rstrip().split("\n"))
    
    # The original full training set is all files except those in validation and test
    original_train_full = set(all_files) - (validation_list | testing_list)
    
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
    print(f"Total files: {len(all_files)}")
    print(f"Original training set: {len(original_train_full)}")
    print(f"Pretraining set: {len(pretraining_set)} samples")
    print(f"Training set: {len(training_set)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", required=True, help="Path to original dataset directory")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to directory with generated lists")
    args = parser.parse_args()
    
    main(args)