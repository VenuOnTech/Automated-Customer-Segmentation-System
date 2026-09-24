import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        # z_i, z_j: [batch_size, projection_dim] (views of the same customer)
        batch_size = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0) # [2B, D]
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Create positive pairs mask
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0)
        labels = labels.to(z_i.device)

        # Mask out self-contrast
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix.masked_fill_(mask, -1e9)

        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss