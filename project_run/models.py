import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForImageClassification

def load_models(config, device):
    # Load tokenizer and pre-trained models
    tokenizer = AutoTokenizer.from_pretrained(config['models']['text_model'])
    text_encoder = AutoModelForSequenceClassification.from_pretrained(config['models']['text_model']).to(device)
    img_encoder  = AutoModelForImageClassification.from_pretrained(config['models']['img_model']).to(device)

    # Remove the classifier of the text encoder
    text_encoder.classifier = nn.Identity().to(device)
    
    # Modify image encoder classifier to match the text encoder's hidden size
    img_hidden_size = img_encoder.config.hidden_size
    txt_hidden_size = text_encoder.config.hidden_size
    new_linear_layer = nn.Linear(img_hidden_size, txt_hidden_size).to(device)
    img_encoder.classifier = new_linear_layer

    return tokenizer, text_encoder, img_encoder
