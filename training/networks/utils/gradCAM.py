import torch
import torch.nn.functional as F

def gradCAM(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Grad-CAM method for visualizing the regions of the image that are important for the model's decision.

    Args:
    - features: torch.Tensor of shape (N, C, H, W) representing the features of the image.
    - weights: torch.Tensor of shape (N, C) representing the weights of the features.

    Returns:
    - torch.Tensor of shape (N, H, W) representing the heatmap of the image.
    """
    N, C, H, W = features.shape
    # Compute the heatmap by taking the weighted sum of the features.
    heatmap = torch.einsum("nchw,nc->nhw", features, weights)
    # Normalize the heatmap.
    heatmap = F.relu(heatmap)
    heatmap /= heatmap.max(dim=(1, 2), keepdim=True)[0]
    return heatmap
    