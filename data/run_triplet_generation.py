import os
import argparse
import numpy as np
import pandas as pd
from TripletGenerator import TripletGenerator

def load_dataset(input_path):
    """Loads the dataset from the provided input path."""
    print(f"Loading dataset from: {input_path}")
    df = pd.read_pickle(input_path)

    # Generate file paths
    df["path"] = df.apply(lambda row: os.path.join(
        "files",
        f"p{str(row.subject_id)[:2]}",
        f"p{str(row.subject_id)}",
        f"s{str(row.study_id)}"
    ), axis=1)

    return df

def multi_hot_encode(labels, mapping):
    """Converts disease labels into a multi-hot encoded vector."""
    vector = np.zeros(len(mapping), dtype=int)
    for label in labels:
        if label in mapping:
            vector[mapping[label]] = 1
    return vector.tolist()

def main(args):
    # Load dataset
    df = load_dataset(args.input)

    # Define disease index mapping
    disease_index_mapping = {
        "pleural other": 0, "consolidation": 1, "lung lesion": 2, "lung opacity": 3,
        "no finding": 4, "enlarged cardiomediastinum": 5, "cardiomegaly": 6, "fracture": 7,
        "pleural effusion": 8, "pneumonia": 9, "edema": 10, "atelectasis": 11, "pneumothorax": 12
    }

    # Apply multi-hot encoding
    print("Applying multi-hot encoding to disease labels...")
    df["disease_labels"] = df["generated_labels"].apply(lambda labels: multi_hot_encode(labels, disease_index_mapping))
    df = df[["path", "generated", "generated_labels", "disease_labels"]]

    # Print hyperparameters
    print(f"Threshold: {args.threshold}")
    print(f"Semi-hard Probability: {args.semi_hard_prob}")
    print(f"Big Batch Size: {args.big_batch_size}")
    print(f"Mini Batch Size: {args.mini_batch_size}")
    print(f"Total Iterations: {args.total_iter}")
    print(f"Save Path: {args.output}")

    # Initialize Triplet Generator
    triplet_generator = TripletGenerator(
        df=df, 
        disease_index_mapping=disease_index_mapping,
        w_dis=args.w_dis, 
        w_adj=args.w_adj, 
        w_dir=args.w_dir
    )

    # Generate triplets
    print("Generating triplets...")
    triplets = triplet_generator.generate_triplets(
        big_batch_size=args.big_batch_size,
        mini_batch_size=args.mini_batch_size,
        total_iter=args.total_iter,
        threshold=args.threshold,
        semi_hard_prob=args.semi_hard_prob
    )

    # Save triplets to CSV
    print(f"Saving generated triplets to: {args.output}")
    triplets_df = pd.DataFrame(triplets)
    triplets_df.to_csv(args.output, index=False, compression='xz')

    print("Triplet generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate triplets from a dataset using OBER-based disease similarity.")

    # Dataset paths
    parser.add_argument('--input', type=str, required=True, help="Path to input pickle file (e.g., /path/to/input.pkl), it should be the output of run_ober.py")
    parser.add_argument('--output', type=str, required=True, help="Path to save the processed triplets CSV (e.g., /path/to/output.csv.xz)")

    # Hyperparameters
    parser.add_argument('--w_dis', type=float, default=0.85, help="Weight for disease similarity (default: 0.85)")
    parser.add_argument('--w_adj', type=float, default=0.10, help="Weight for adjective similarity (default: 0.10)")
    parser.add_argument('--w_dir', type=float, default=0.05, help="Weight for direction similarity (default: 0.05)")

    parser.add_argument('--threshold', type=float, default=0.25, help="Threshold for negative sample selection (default: 0.25)")
    parser.add_argument('--semi_hard_prob', type=float, default=1.0, help="Probability of choosing a semi-hard negative (default: 1.0)")

    parser.add_argument('--big_batch_size', type=int, default=512, help="Size of the big batch (default: 512)")
    parser.add_argument('--mini_batch_size', type=int, default=32, help="Size of the mini-batch (default: 32)")
    parser.add_argument('--total_iter', type=int, default=40000, help="Total number of triplet generation iterations (default: 40000)")

    args = parser.parse_args()
    main(args)