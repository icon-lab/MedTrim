import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

class TripletGenerator:
    """
    A class for generating triplets (anchor, positive, negative) for metric learning.

    This class calculates similarity between medical image labels using:
    - Jaccard similarity for disease labels
    - Intersection-over-Union (IoU) for adjectives and lung directions

    The generated triplets can be used for training models such as contrastive learning 
    and triplet loss-based architectures.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image paths, disease labels, and ontology-based extracted attributes.
    disease_index_mapping : dict
        Dictionary mapping disease names to index values for multi-hot encoding.
    w_dis : float, optional
        Weight for disease label similarity in scoring (default: 0.85).
    w_adj : float, optional
        Weight for adjective similarity in scoring (default: 0.10).
    w_dir : float, optional
        Weight for lung direction similarity in scoring (default: 0.05).
    """

    def __init__(self, df, disease_index_mapping, w_dis=0.85, w_adj=0.10, w_dir=0.05):
        """Initializes the TripletGenerator with dataset and similarity weights."""
        self.df = df
        self.disease_index_mapping = disease_index_mapping
        self.w_dis = w_dis
        self.w_adj = w_adj
        self.w_dir = w_dir

    @staticmethod
    def calculate_jaccard_score(anchor, reference):
        """Computes the Jaccard similarity between two multi-hot encoded disease label vectors."""
        intersection = np.logical_and(anchor, reference).sum()
        union = np.logical_or(anchor, reference).sum()
        return intersection / union if union != 0 else 1

    @staticmethod
    def calculate_iou(set1, set2):
        """Computes the Intersection-over-Union (IoU) between two sets."""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 1

    def calculate_score(self, anchor_dict, ref_dict, key, weight):
        """
        Computes weighted IoU score for a specific key (adjectives or lung directions).

        Parameters
        ----------
        anchor_dict : dict
            Dictionary of extracted attributes for anchor sample.
        ref_dict : dict
            Dictionary of extracted attributes for reference sample.
        key : str
            Key indicating whether to compute score for 'adjs' or 'dirs'.
        weight : float
            Weight applied to the computed score.

        Returns
        -------
        float
            Weighted similarity score.
        """
        common_sicknesses = set(anchor_dict.keys()) & set(ref_dict.keys())
        iou_sum = sum(self.calculate_iou(set(anchor_dict[s][key]), set(ref_dict[s][key])) for s in common_sicknesses)
        return (iou_sum / len(anchor_dict)) * weight if anchor_dict else 0

    def calculate_adjective_score(self, anchor_dict, ref_dict):
        """Computes adjective similarity score using IoU-based weighting."""
        return self.calculate_score(anchor_dict, ref_dict, 'adjs', self.w_adj)

    def calculate_direction_score(self, anchor_dict, ref_dict):
        """Computes lung direction similarity score using IoU-based weighting."""
        return self.calculate_score(anchor_dict, ref_dict, 'dirs', self.w_dir)

    def calculate_total_score(self, anchor_idx, reference_idx):
        """
        Computes the total similarity score between an anchor and a reference sample.

        Parameters
        ----------
        anchor_idx : int
            Index of the anchor sample in the DataFrame.
        reference_idx : int
            Index of the reference sample in the DataFrame.

        Returns
        -------
        float
            Combined similarity score based on disease, adjective, and lung direction similarities.
        """
        anchor_disease_labels = self.df.loc[anchor_idx, 'disease_labels']
        reference_disease_labels = self.df.loc[reference_idx, 'disease_labels']

        jaccard_score = self.calculate_jaccard_score(anchor_disease_labels, reference_disease_labels) * self.w_dis

        anchor_generated = self.df.loc[anchor_idx, 'generated']
        reference_generated = self.df.loc[reference_idx, 'generated']

        adjective_score = self.calculate_adjective_score(anchor_generated, reference_generated)
        direction_score = self.calculate_direction_score(anchor_generated, reference_generated)

        return jaccard_score + adjective_score + direction_score

    def APNFinder(self, anchor_idx, df_mini, threshold, semi_hard_prob):
        """
        Finds the positive and negative samples for a given anchor.

        Parameters
        ----------
        anchor_idx : int
            Index of the anchor sample.
        df_mini : pandas.DataFrame
            Subset of the DataFrame to search for triplets.
        threshold : float
            Score threshold for selecting semi-hard negatives.
        semi_hard_prob : float
            Probability of choosing a semi-hard negative sample.

        Returns
        -------
        tuple
            (positive_idx, negative_idx, positive_score, negative_score)
        """
        scores = [(idx, self.calculate_total_score(anchor_idx, idx)) for idx in df_mini.index if idx != anchor_idx]
        scores.sort(key=lambda x: x[1], reverse=True)

        positive_idx, positive_score = scores[0]
        scores = scores[1:]

        scores = [(idx, score) for idx, score in scores if score < 0.6]

        if random.random() < semi_hard_prob:
            negative_candidates = [(idx, score) for idx, score in scores if score >= threshold]
            negative_idx, negative_score = (
                min(negative_candidates, key=lambda x: x[1]) if negative_candidates else max(scores, key=lambda x: x[1])
            )
        else:
            negative_idx, negative_score = scores[-1]

        return positive_idx, negative_idx, positive_score, negative_score

    def generate_triplets(self, big_batch_size, mini_batch_size, total_iter, threshold, semi_hard_prob):
        """
        Generates triplets (anchor, positive, negative) based on disease label similarities.

        Parameters
        ----------
        big_batch_size : int
            Number of samples to consider in each batch.
        mini_batch_size : int
            Number of samples per mini-batch.
        total_iter : int
            Total number of triplets to generate.
        threshold : float
            Threshold for selecting semi-hard negatives.
        semi_hard_prob : float
            Probability of selecting a semi-hard negative.

        Returns
        -------
        list of dict
            List of triplets, each containing paths of anchor, positive, and negative samples.
        """
        triplets = []
        for _ in tqdm(range(total_iter), desc="Creating Triplets"):
            selected_indices = random.sample(range(len(self.df)), big_batch_size)

            for _ in range(mini_batch_size):
                mini_batch_indices = random.sample(selected_indices, mini_batch_size)
                random_anchor = random.choice(mini_batch_indices)

                anchor_idx = self.df.iloc[mini_batch_indices].index[mini_batch_indices.index(random_anchor)]
                positive_idx, negative_idx, _, _ = self.APNFinder(anchor_idx, self.df.iloc[mini_batch_indices], threshold, semi_hard_prob)

                triplets.append({
                    'anchor': self.df.loc[anchor_idx, "path"],
                    'positive': self.df.loc[positive_idx, "path"],
                    'negative': self.df.loc[negative_idx, "path"]
                })

        return triplets
