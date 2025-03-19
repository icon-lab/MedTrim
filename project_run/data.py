import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def load_and_prepare_data(config, tokenizer):
    # Load CSV files
    img_df = pd.read_csv(config['data']['img_df'])
    text_df = pd.read_csv(config['data']['text_df'])
    triplet_df = pd.read_csv(config['data']['triplet_csv'], compression="infer")
    
    # Create lookup dictionaries
    img_path_to_full = dict(zip(img_df["path_base"], img_df["full_path"]))
    text_path_to_text = dict(zip(text_df["path_base"], text_df["text"]))
    
    # Sample triplets (here sampling 2 million examples; adjust if needed)
    sampled_triplet_df = triplet_df.sample(n=2000000, random_state=config['training']['random_seed'])
    
    data_dicts = []
    for _, row in tqdm(sampled_triplet_df.iterrows(), desc="Preparing data"):
        a_path, p_path, n_path = row["anchor"], row["positive"], row["negative"]
        try:
            img_a_path = img_path_to_full[a_path]
            img_p_path = img_path_to_full[p_path]
            img_n_path = img_path_to_full[n_path]
            text_a = text_path_to_text.get(a_path, '')
            text_p = text_path_to_text.get(p_path, '')
            text_n = text_path_to_text.get(n_path, '')
            
            dict_elem = {
                "img_a": img_a_path,
                "img_p": img_p_path,
                "img_n": img_n_path,
                "text_a": text_a,
                "text_p": text_p,
                "text_n": text_n
            }
            data_dicts.append(dict_elem)
        except KeyError:
            continue

    # Shuffle the data for randomness
    random.seed(config['training']['random_seed'])
    random.shuffle(data_dicts)
    
    # Split into train and validation sets (80/10 split, remaining could be used as test)
    train_size = int(len(data_dicts) * 0.8)
    val_size = int(len(data_dicts) * 0.1)
    train_data = data_dicts[:train_size]
    val_data = data_dicts[train_size:train_size + val_size]
    
    # Preprocess text fields (tokenization)
    train_data_processed = preprocess_text(train_data, tokenizer)
    val_data_processed = preprocess_text(val_data, tokenizer)
    
    return train_data_processed, val_data_processed


def preprocess_text(data_list, tokenizer):
    processed_data = []
    for item in tqdm(data_list, desc="Tokenizing texts"):
        processed_item = {}
        for key, value in item.items():
            if key.startswith("text_"):
                tokenized = tokenizer(value, truncation=True, padding="max_length", max_length=128)
                # Convert tokenization output (a dict) into a NumPy array and reshape
                processed_item[key] = np.array(list(tokenized.values())).reshape(3, -1)
            else:
                processed_item[key] = value
        processed_data.append(processed_item)
    return processed_data
