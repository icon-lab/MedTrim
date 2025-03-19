def freeze_first_n_layers(model, n):
    """
    Freeze the first n layers of the encoder.
    Supports models with either a BERT or ViT architecture.
    """
    if hasattr(model, 'bert'):
        layers = model.bert.encoder.layer
    elif hasattr(model, 'vit'):
        layers = model.vit.encoder.layer
    else:
        raise ValueError("Unknown model architecture for freezing layers.")
        
    for i, layer in enumerate(layers):
        if i < n:
            for param in layer.parameters():
                param.requires_grad = False


def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params
