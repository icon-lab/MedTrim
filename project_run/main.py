import os
import yaml
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data import load_and_prepare_data
from dataset import TripletDataset
from models import load_models
from train import train_epoch, validate_epoch
from loss import TripletLoss
from utils import freeze_first_n_layers, count_parameters

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'])

    # Load models and tokenizer
    tokenizer, text_encoder, img_encoder = load_models(config, device)

    # Freeze the first 2 layers of both encoders
    freeze_first_n_layers(text_encoder, 2)
    freeze_first_n_layers(img_encoder, 2)

    # Print parameter counts for debugging
    text_trainable, text_non_trainable = count_parameters(text_encoder)
    print(f"Text Encoder - Trainable: {text_trainable}, Non-trainable: {text_non_trainable}")
    img_trainable, img_non_trainable = count_parameters(img_encoder)
    print(f"Image Encoder - Trainable: {img_trainable}, Non-trainable: {img_non_trainable}")

    # Load and preprocess data
    train_data, val_data = load_and_prepare_data(config, tokenizer)

    # Define image transformations
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset objects
    train_dataset = TripletDataset(train_data, img_transform=img_transform)
    val_dataset = TripletDataset(val_data, img_transform=img_transform)

    # Build dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=config['training']['batch_size'],
                              shuffle=True,
                              num_workers=config['training']['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=False,
                            num_workers=config['training']['num_workers'],
                            pin_memory=True)

    # Setup loss, optimizers, and gradient scaler
    triplet_loss = TripletLoss(margin=config['models']['margin'])
    scaler = torch.cuda.amp.GradScaler()

    optimizer_img = torch.optim.AdamW(img_encoder.parameters(),
                                        lr=config['training']['learning_rate'],
                                        weight_decay=config['training']['weight_decay'])
    optimizer_text = torch.optim.AdamW(text_encoder.parameters(),
                                         lr=config['training']['learning_rate'],
                                         weight_decay=config['training']['weight_decay'])

    num_epochs = config['training']['num_epochs']
    model_save_path = config['data']['model_save_path']
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_img_loss, train_text_loss = train_epoch(train_loader,
                                                      text_encoder,
                                                      img_encoder,
                                                      optimizer_text,
                                                      optimizer_img,
                                                      scaler,
                                                      triplet_loss,
                                                      device)
        val_img_loss, val_text_loss = validate_epoch(val_loader,
                                                     text_encoder,
                                                     img_encoder,
                                                     triplet_loss,
                                                     device)

        print(f"Train - Image Loss: {train_img_loss:.4f}, Text Loss: {train_text_loss:.4f}")
        print(f"Val   - Image Loss: {val_img_loss:.4f}, Text Loss: {val_text_loss:.4f}")

        # Save models periodically
        if (epoch + 1) % config['training']['save_freq'] == 0 or (epoch + 1) == num_epochs:
            torch.save(img_encoder.state_dict(), f"{model_save_path}/img_encoder_epoch_{epoch + 1}.pth")
            torch.save(text_encoder.state_dict(), f"{model_save_path}/text_encoder_epoch_{epoch + 1}.pth")
            print(f"Saved models at epoch {epoch + 1}")

if __name__ == "__main__":
    main()
