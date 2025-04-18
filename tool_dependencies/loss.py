import torch


def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)

def cosine_similarity(x, y):
    """Cosine similarity."""
    return torch.nn.CosineSimilarity(dim=1, eps=1e-6)(x, y)


def approximate_hamming_similarity(x, y):
    """Approximate Hamming similarity."""
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


def pairwise_loss(x, y, labels, loss_type='margin', margin=1.0):
    """Compute pairwise loss.

    Args:
      x: [N, D] float tensor, representations for N examples.
      y: [N, D] float tensor, representations for another N examples.
      labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
        and y[i] are similar, and -1 otherwise.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """

    labels = labels.float()

    if loss_type == 'margin':
        return torch.relu((margin - labels * (1 - euclidean_distance(x, y))).detach())
    elif loss_type == 'hamming':
        return 0.25 * (approximate_hamming_similarity(x, y) - labels) ** 2
    elif loss_type == 'cosine':
        return 0.25 * (cosine_similarity(x, y) - labels) ** 2
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)