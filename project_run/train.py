# train.py
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from loss import TripletLoss  

def train_epoch(train_loader, text_encoder, img_encoder, optimizer_text, optimizer_img, scaler, triplet_loss, device):
    img_train_loss, text_train_loss = 0.0, 0.0
    for batch in tqdm(train_loader, desc="Training"):
        # Move image data to device
        img_a = batch["img_a"].to(device)
        img_p = batch["img_p"].to(device)
        img_n = batch["img_n"].to(device)
        # Move text data to device
        text_a = batch["text_a"].to(device)
        text_p = batch["text_p"].to(device)
        text_n = batch["text_n"].to(device)

        optimizer_img.zero_grad()
        optimizer_text.zero_grad()

        with autocast():
            # Forward pass through image encoder
            img_a_out = img_encoder(img_a).logits
            img_p_out = img_encoder(img_p).logits
            img_n_out = img_encoder(img_n).logits

            # Forward pass through text encoder
            text_a_out = text_encoder(input_ids=text_a[:, 0, :], attention_mask=text_a[:, 2, :]).logits
            text_p_out = text_encoder(input_ids=text_p[:, 0, :], attention_mask=text_p[:, 2, :]).logits
            text_n_out = text_encoder(input_ids=text_n[:, 0, :], attention_mask=text_n[:, 2, :]).logits

            loss_img, loss_text = triplet_loss(img_a_out, img_p_out, img_n_out,
                                               text_a_out, text_p_out, text_n_out)
            total_loss = loss_img + loss_text

        scaler.scale(total_loss).backward()
        scaler.step(optimizer_img)
        scaler.step(optimizer_text)
        scaler.update()

        img_train_loss += loss_img.item()
        text_train_loss += loss_text.item()

    avg_img_loss = img_train_loss / len(train_loader)
    avg_text_loss = text_train_loss / len(train_loader)
    return avg_img_loss, avg_text_loss


def validate_epoch(val_loader, text_encoder, img_encoder, triplet_loss, device):
    img_val_loss, text_val_loss = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            img_a = batch["img_a"].to(device)
            img_p = batch["img_p"].to(device)
            img_n = batch["img_n"].to(device)
            text_a = batch["text_a"].to(device)
            text_p = batch["text_p"].to(device)
            text_n = batch["text_n"].to(device)

            img_a_out = img_encoder(img_a).logits
            img_p_out = img_encoder(img_p).logits
            img_n_out = img_encoder(img_n).logits

            text_a_out = text_encoder(input_ids=text_a[:, 0, :], attention_mask=text_a[:, 2, :]).logits
            text_p_out = text_encoder(input_ids=text_p[:, 0, :], attention_mask=text_p[:, 2, :]).logits
            text_n_out = text_encoder(input_ids=text_n[:, 0, :], attention_mask=text_n[:, 2, :]).logits

            loss_img, loss_text = triplet_loss(img_a_out, img_p_out, img_n_out,
                                               text_a_out, text_p_out, text_n_out)
            img_val_loss += loss_img.item()
            text_val_loss += loss_text.item()
    avg_img_loss = img_val_loss / len(val_loader)
    avg_text_loss = text_val_loss / len(val_loader)
    return avg_img_loss, avg_text_loss
