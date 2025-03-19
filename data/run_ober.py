import pickle
import argparse
import pandas as pd
from OBER import OBER

def main(input_path, output_path):
    # Load dataset
    # df should be consist of columns: subject_id, study_id, content, findings, impression, label
    df = pd.read_csv(input_path, compression='gzip', converters={'labels': pd.eval}).fillna({'findings': "", 'impressions': ""})
    
    # Define ontology and mappings (Modify as needed)
    file_path = "data/Ontology.pkl" 
    # Load the Pickle file
    with open(file_path, "rb") as f:
        ontology_data = pickle.load(f)
    
    keys_to_keep = {'ADJECTIVE', 'LUNG_DIRECTION', 'DISEASE'}

    ontology = {key: ontology_data[key] for key in keys_to_keep if key in ontology_data}
    disease_mapping = ontology_data["DISEASE_MAPPING"]
    adjective_mapping = ontology_data["ADJECTIVE_MAPPING"]

    # Initialize OBER instance
    ober = OBER(ontology, disease_mapping, adjective_mapping)

    # Filter non-empty findings/impressions
    df_filtered = df[(df['findings'] != '') | (df['impressions'] != '')]

    # Apply OBER processing
    df_filtered['generated'] = df_filtered.apply(lambda row: OBER.extract_labels(row, ober), axis=1)

    # Post-process labels
    df_filtered['labels'] = df_filtered['labels'].apply(OBER.modify_list)
    df_filtered["generated_labels"] = df_filtered["generated"].apply(lambda lst: [key for d in lst for key in d.keys()])
    df_filtered['generated'] = df_filtered['generated'].apply(OBER.list_to_dict)

    # Ensure 'no finding' is included when necessary
    df_filtered['labels'] = df_filtered['labels'].apply(OBER.add_no_finding)
    df_filtered['generated_labels'] = df_filtered['generated_labels'].apply(OBER.add_no_finding)

    # Sort labels and generated labels
    df_filtered['generated_labels'] = df_filtered['generated_labels'].apply(OBER.sort_list)
    df_filtered['labels'] = df_filtered['labels'].apply(OBER.sort_list)
    df_filtered['generated'] = df_filtered['generated'].apply(OBER.sort_dict_by_key)

    # Save the processed DataFrame
    df_filtered.to_pickle(output_path)

    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply OBER processing on a dataset.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the processed CSV file")
    args = parser.parse_args()

    main(args.input, args.output)
