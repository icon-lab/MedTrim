import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Ensure required NLTK resources are downloaded
nltk.download('punkt')

class OBER:
    def __init__(self, ontology, disease_mapping=None, adjective_mapping=None):
        """
        Initialize the OBER object with ontology and optional mappings.

        :param ontology: Dictionary containing categorized terms
        :param disease_mapping: Mapping dictionary for disease terms (optional)
        :param adjective_mapping: Mapping dictionary for adjective terms (optional)
        """
        self.ontology = ontology
        self.disease_mapping = disease_mapping or {}
        self.adjective_mapping = adjective_mapping or {}

    def ontology_based_ner(self, sentence):
        """Performs Named Entity Recognition (NER) using ontology."""
        detected_entities = {key: set() for key in self.ontology.keys()}

        for entity_type, entity_list in self.ontology.items():
            sorted_entities = sorted(entity_list, key=len, reverse=True)
            for entity in sorted_entities:
                if f" {entity} " in sentence:
                    if entity_type == "DELETE_SENTENCE":
                        return {key: set() for key in self.ontology.keys()}
                    detected_entities[entity_type].add(entity)
                    sentence = sentence.replace(entity, "")

        return detected_entities

    def process_text(self, text):
        """Tokenizes text and extracts entities based on ontology."""
        sentences = sent_tokenize(text)
        results = []
        parts = []

        for sentence in sentences:
            for part in re.split(r'\b(?:but|with|, which)\b', sentence):
                part = part.split(" without ")[0].split(" rather than ")[0]
                part = " " + re.sub(r'[^a-zA-Z]', ' ', part)
                parts.append(part)
                results.append(self.ontology_based_ner(part))

        return results

    def get_registered_labels(self, text):
        """Extracts and organizes detected entities into structured labels."""
        results = self.process_text(text)
        labels = {self.disease_mapping.get(dis, dis): {"adjs": set(), "dirs": set()} 
                  for result in results for dis in result["DISEASE"]}

        for result in results:
            for dis in result["DISEASE"]:
                disease = self.disease_mapping.get(dis, dis)
                labels[disease]["adjs"].update(
                    {self.adjective_mapping.get(adj, adj) for adj in result["ADJECTIVE"]}
                )
                labels[disease]["dirs"].update(result["LUNG_DIRECTION"])

        return labels

    def apply_ontology(self, row):
        """Applies ontology-based entity extraction on a DataFrame."""
        text = (row.get('findings', '') + " " + row.get('impressions', '')).strip()
        if not text:
            raise ValueError("Either 'findings' or 'impressions' must be non-empty.")
        return self.get_registered_labels(text)

    @staticmethod
    def add_labels(row, raw_ontology='dataset1'):
        """Adds missing labels to ontology."""
        ontology = row[raw_ontology].copy()
        for label in row['labels']:
            if label not in list(ontology.keys()) + ['no finding', 'support devices']:
                ontology[label] = {"adjs": set(), "dirs": set()}
        return ontology

    @staticmethod
    def extract_labels(row, ober_instance):
        """Extracts labels using an OBER instance."""
        return [ober_instance.apply_ontology(row)]

    @staticmethod
    def add_no_finding(label_list):
        """Ensures 'no finding' label is added if no labels exist."""
        return label_list if label_list else ['no finding']

    @staticmethod
    def modify_list(label_list):
        """Removes 'support devices' label from the list."""
        if "support devices" in label_list:
            label_list.remove("support devices")
        return label_list

    @staticmethod
    def sort_list(label_list):
        """Sorts label list alphabetically."""
        return sorted(label_list)

    @staticmethod
    def sort_dict_by_key(d):
        """Sorts dictionary by keys."""
        return dict(sorted(d.items()))

    @staticmethod
    def list_to_dict(lst):
        """Converts list to a dictionary by taking the first element."""
        return lst[0] if lst else {}
