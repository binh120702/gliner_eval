import torch
import torch.nn.functional as F


def focal_loss_with_logits(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        prob_margin: float = 0.,
        reduction: str = "none",
        label_smoothing: float = 0.0,
        ignore_index: int = -100,  # default value for ignored index
        eps: float = 1e-6
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
        label_smoothing (float): Specifies the amount of smoothing when computing the loss, 
                                                                where 0.0 means no smoothing.
        ignore_index (int): Specifies a target value that is ignored and does not contribute
                            to the input gradient. Default: ``-100``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Create a mask to ignore specified index
    valid_mask = targets != ignore_index
    
    # Apply label smoothing if needed
    if label_smoothing != 0:
        with torch.no_grad():
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    # Apply sigmoid activation to inputs
    p = torch.sigmoid(inputs)

    pm =  torch.clamp(p-prob_margin, max=1.0)

    # Compute the binary cross-entropy loss without reduction
    pos_term = -targets * torch.log(p.clamp(min=eps))
    neg_term = -(1.0 - targets) * torch.log((1.0 - pm).clamp(min=eps))
    loss = (pos_term + neg_term)    
    # F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Apply the valid mask to the loss
    loss = loss * valid_mask

    # Apply focal loss modulation if gamma is greater than 0
    if gamma > 0:
        p_t = p * targets + (1-pm) * (1 - targets)
        loss = loss * ((1 - p_t) ** gamma)

    # Apply alpha weighting if alpha is specified
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Apply reduction method
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.sum() / valid_mask.sum()  # Normalize by the number of valid (non-ignored) elements
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(
            f"Invalid value for argument 'reduction': '{reduction}'. "
            f"Supported reduction modes: 'none', 'mean', 'sum'"
        )


def cross_entropy_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "sum",
        label_smoothing: float = 0.0,
        ignore_index: int = -100,  # default value for ignored index
        **kwargs
) -> torch.Tensor:

    cls_size = inputs.shape[-1]
    inputs = inputs.reshape(-1, cls_size)
    targets = targets.reshape(-1)
    loss = F.cross_entropy(inputs, targets, ignore_index = ignore_index, 
                            label_smoothing = label_smoothing, reduction=reduction)

    return loss


def adversarial_contrastive_loss(
        span_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.1,
        margin: float = 0.5,
        confusion_threshold: float = 0.3,
        reduction: str = "mean",
        eps: float = 1e-6
) -> torch.Tensor:
    """
    Adversarial contrastive loss that pushes span embeddings away from wrong entity embeddings,
    especially for close/confusing entity pairs.
    
    Args:
        span_embeddings: Span representations [B, L, W, D] or [B*L*W, D]
        entity_embeddings: Entity/prompt representations [B, C, D] or [B*C, D]
        labels: Binary labels [B, L*W, C] or [B*L*W, C]
        temperature: Temperature parameter for contrastive learning
        margin: Margin for hard negative mining
        confusion_threshold: Threshold to identify confusing entity pairs
        reduction: Reduction method ('mean', 'sum', 'none')
        eps: Small value for numerical stability
    
    Returns:
        Contrastive loss tensor
    """
    # Normalize embeddings
    span_embeddings = F.normalize(span_embeddings, p=2, dim=-1)
    entity_embeddings = F.normalize(entity_embeddings, p=2, dim=-1)
    
    # Handle different input shapes
    if span_embeddings.dim() == 4:  # [B, L, W, D]
        B, L, W, D = span_embeddings.shape
        span_embeddings = span_embeddings.view(B, L*W, D)
    elif span_embeddings.dim() == 3:  # [B, L*W, D]
        B, LW, D = span_embeddings.shape
        L, W = 1, LW  # Treat as single dimension
    else:  # [B*L*W, D]
        B = 1
        L, W = 1, span_embeddings.shape[0]
        D = span_embeddings.shape[1]
        span_embeddings = span_embeddings.unsqueeze(0)
    
    if entity_embeddings.dim() == 3:  # [B, C, D]
        B_ent, C, D_ent = entity_embeddings.shape
        if B_ent != B:
            raise ValueError(f"Batch size mismatch: span_embeddings {B}, entity_embeddings {B_ent}")
    else:  # [B*C, D]
        B_ent = B
        C = entity_embeddings.shape[0] // B
        D_ent = entity_embeddings.shape[1]
        entity_embeddings = entity_embeddings.view(B, C, D_ent)
    
    if D != D_ent:
        raise ValueError(f"Embedding dimension mismatch: span_embeddings {D}, entity_embeddings {D_ent}")
    
    # Ensure labels have the right shape [B, L*W, C]
    if labels.dim() == 2:  # [B*L*W, C]
        labels = labels.view(B, L*W, C)
    elif labels.dim() == 3:  # [B, L*W, C]
        pass  # Already correct shape
    else:
        raise ValueError(f"Invalid labels shape: {labels.shape}")
    
    total_loss = 0.0
    valid_samples = 0
    
    # Process each batch
    for b in range(B):
        # Get span and entity embeddings for this batch
        batch_spans = span_embeddings[b]  # [L*W, D]
        batch_entities = entity_embeddings[b]  # [C, D]
        batch_labels = labels[b]  # [L*W, C]
        
        # Compute similarity matrix for this batch
        # [L*W, D] @ [D, C] -> [L*W, C]
        similarities = torch.matmul(batch_spans, batch_entities.T) / temperature
        
        # Process each span
        for s in range(L*W):
            span_similarities = similarities[s]  # [C]
            span_labels = batch_labels[s]  # [C]
            
            # Get positive and negative entities for this span
            pos_mask = span_labels > 0.5  # Positive entities
            neg_mask = span_labels <= 0.5  # Negative entities
            
            if pos_mask.sum() == 0:  # Skip if no positive entities
                continue
                
            # Positive loss: maximize similarity with correct entities
            pos_similarities = span_similarities[pos_mask]
            if len(pos_similarities) > 0:
                pos_loss = -torch.log(torch.sigmoid(pos_similarities).clamp(min=eps)).mean()
            else:
                pos_loss = 0.0
            
            # Negative loss: minimize similarity with wrong entities
            neg_similarities = span_similarities[neg_mask]
            if len(neg_similarities) > 0:
                # Identify confusing negatives (high similarity with wrong entities)
                confusing_mask = torch.sigmoid(neg_similarities) > confusion_threshold
                
                if confusing_mask.sum() > 0:
                    # Hard negative mining: focus on confusing entities
                    hard_neg_similarities = neg_similarities[confusing_mask]
                    neg_loss = torch.log(torch.sigmoid(-hard_neg_similarities + margin).clamp(min=eps)).mean()
                else:
                    # Standard negative loss
                    neg_loss = torch.log(torch.sigmoid(-neg_similarities).clamp(min=eps)).mean()
            else:
                neg_loss = 0.0
            
            # Combine positive and negative losses
            span_loss = pos_loss + neg_loss
            total_loss += span_loss
            valid_samples += 1
    
    if valid_samples == 0:
        return torch.tensor(0.0, device=span_embeddings.device, requires_grad=True)
    
    if reduction == "mean":
        return total_loss / valid_samples
    elif reduction == "sum":
        return total_loss
    elif reduction == "none":
        return total_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def combined_adversarial_loss(
        scores: torch.Tensor,
        labels: torch.Tensor,
        span_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        contrastive_weight: float = 0.1,
        temperature: float = 0.1,
        margin: float = 0.5,
        confusion_threshold: float = 0.3,
        reduction: str = "mean",
        **kwargs
) -> torch.Tensor:
    """
    Combined loss that integrates focal loss with adversarial contrastive loss.
    
    Args:
        scores: Model predictions [B, L*W, C]
        labels: Ground truth labels [B, L*W, C]
        span_embeddings: Span representations [B, L, W, D]
        entity_embeddings: Entity representations [B, C, D]
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        contrastive_weight: Weight for contrastive loss component
        temperature: Temperature for contrastive loss
        margin: Margin for hard negative mining
        confusion_threshold: Threshold for identifying confusing pairs
        reduction: Reduction method
        **kwargs: Additional arguments for focal loss
    
    Returns:
        Combined loss tensor
    """
    # Compute focal loss
    focal_loss = focal_loss_with_logits(
        scores, labels,
        alpha=focal_alpha,
        gamma=focal_gamma,
        reduction=reduction,
        **kwargs
    )
    
    # Compute adversarial contrastive loss
    contrastive_loss = adversarial_contrastive_loss(
        span_embeddings, entity_embeddings, labels,
        temperature=temperature,
        margin=margin,
        confusion_threshold=confusion_threshold,
        reduction=reduction
    )
    
    # Combine losses
    total_loss = focal_loss + contrastive_weight * contrastive_loss
    
    return total_loss