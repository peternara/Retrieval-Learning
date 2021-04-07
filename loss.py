import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    ''' Implements contrastive loss to train Siamese Network'''

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_1, output_2, label):
        # Distance between embedded outputs
        L2_distance = F.pairwise_distance(output_1, output_2, keepdim = True)
        
        # Loss calculation
        loss_contrastive = torch.mean((1-label) * torch.pow(L2_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - L2_distance, min=0.0), 2))

        return loss_contrastive

def contrastive_batch_loss(output_1, output_2, label, margin=2.0 ):
    ''' Calculates individual contrastive loss for each element in the batch'''

    # Distance between embedded outputs
    L2_distance = F.pairwise_distance(output_1, output_2, keepdim = True)

    # Loss calculation
    loss = (1-label) * torch.pow(L2_distance, 2) + (label) * torch.pow(torch.clamp(margin - L2_distance, min=0.0), 2)
    
    return loss