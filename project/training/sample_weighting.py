"""
Sample Weighting — Applies confidence-based weights to the training loss.
"""
import torch

def compute_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute binary cross entropy loss weighted by negative confidence sample weights.

    Args:
        logits: model raw logits (shape: [batch_size])
        labels: true labels (shape: [batch_size], values: 0.0 or 1.0)
        sample_weights: sample weights (shape: [batch_size], values: 0.0 to 1.0)
        pos_weight: optional class weighting for positive class

    Returns:
        weighted loss scalar
    """
    # Use BCEWithLogitsLoss with reduction='none' to get individual loss values
    loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([pos_weight], device=logits.device))
    raw_losses = loss_fct(logits, labels.float())

    # Multiply by confidence-based sample weights
    weighted_losses = raw_losses * sample_weights.float()

    # Sum of weights in batch to normalize correctly
    weight_sum = sample_weights.sum()
    if weight_sum > 1e-5:
        return weighted_losses.sum() / weight_sum
    else:
        return weighted_losses.mean()
