# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, a_i, p_i, n_i, a_t, p_t, n_t):
        # Compute cosine similarities and derive distances
        distance_i_1 = 1 - F.cosine_similarity(a_i, p_i)
        distance_i_2 = 1 - F.cosine_similarity(a_i, n_i)
        
        distance_t_1 = 1 - F.cosine_similarity(a_t, p_t)
        distance_t_2 = 1 - F.cosine_similarity(a_t, n_t)
        
        distance_it_1 = 1 - F.cosine_similarity(a_i, p_t)
        distance_it_2 = 1 - F.cosine_similarity(a_i, n_t)
        
        distance_ti_1 = 1 - F.cosine_similarity(a_t, p_i)
        distance_ti_2 = 1 - F.cosine_similarity(a_t, n_i)

        loss_im_enc_ii = torch.clamp(distance_i_1 - distance_i_2 + self.margin, min=0)
        loss_im_enc_it = torch.clamp(distance_it_1 - distance_it_2 + self.margin, min=0)
        loss_im_enc = (loss_im_enc_ii + loss_im_enc_it) / 2.0

        loss_text_enc_tt = torch.clamp(distance_t_1 - distance_t_2 + self.margin, min=0)
        loss_text_enc_ti = torch.clamp(distance_ti_1 - distance_ti_2 + self.margin, min=0)
        loss_text_enc = (loss_text_enc_tt + loss_text_enc_ti) / 2.0

        return torch.mean(loss_im_enc), torch.mean(loss_text_enc)
