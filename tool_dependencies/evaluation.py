from sklearn import metrics
from tool_dependencies.loss import *

def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)


def compute_similarity(config, x, y):
    """Compute the distance between x and y vectors.

    The distance will be computed based on the training loss type.

    Args:
      config: a config dict.
      x: [n_examples, feature_dim] float tensor.
      y: [n_examples, feature_dim] float tensor.

    Returns:
      dist: [n_examples] float tensor.

    Raises:
      ValueError: if loss type is not supported.
    """
    if config['training']['loss'] == 'margin':
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif config['training']['loss'] == 'hamming':
        return exact_hamming_similarity(x, y)
    elif config['training']['loss'] == 'cosine':
        return cosine_similarity(x, y)
    else:
        raise ValueError('Unknown loss type %s' % config['training']['loss'])


def auc(scores, labels, **auc_args):
    """Compute the AUC for pair classification.

    See `tf.metrics.auc` for more details about this metric.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1. (Hamming & Euclidean), positive scores mean higher preference
        of being assigned to label +1 while negative scores mean higher preference of being assigned
        to label -1. (Cosine) scores are setted to +1 or -1 based on the treshold imposed.
      labels: [n_examples] int.  Labels are either +1 or -1.

    Returns:
      auc: the area under the ROC curve.
    """

    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    # scale labels from [-1,1] to [0,1]
    labels = (labels + 1) / 2

    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return metrics.auc(fpr, tpr)

def f1score(scores,labels,similarity_type, threshold):
    """Compute the F1-score for pair classification.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1. (Hamming & Euclidean), positive scores mean higher preference
        of being assigned to label +1 while negative scores mean higher preference of being assigned
        to label -1. (Cosine) scores are setted to +1 or -1 based on the treshold imposed.
      labels: [n_examples] int.  Labels are either +1 or -1.
      similarity_type: the type of similarity used to compute the scores.
      threshold: the threshold used to binarize the scores.

    Returns:
      f1: the F1-score.
    """

    if similarity_type == 'cosine':
      scores = torch.sign(scores + 1e-8 - threshold)
    
    else:
      scores_max = torch.max(scores)
      scores_min = torch.min(scores)

      # normalize scores to [0, 1] and add a small epislon for safety
      scores = torch.round((scores - scores_min) / (scores_max - scores_min + 1e-8))

      # scale labels from [-1,1] to [0,1]
      labels = (labels + 1) / 2

    return metrics.f1_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())

def precision(scores,labels,similarity_type, threshold):
    """Compute the precision for pair classification.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1. (Hamming & Euclidean), positive scores mean higher preference
        of being assigned to label +1 while negative scores mean higher preference of being assigned
        to label -1. (Cosine) scores are setted to +1 or -1 based on the treshold imposed.
      labels: [n_examples] int.  Labels are either +1 or -1.
      similarity_type: the type of similarity used to compute the scores.
      threshold: the threshold used to binarize the scores.

    Returns:
      precision: the precision score.
    """

    if similarity_type == 'cosine':
      scores = torch.sign(scores + 1e-8 - threshold)
    
    else:
      scores_max = torch.max(scores)
      scores_min = torch.min(scores)

      # normalize scores to [0, 1] and add a small epislon for safety
      scores = torch.round((scores - scores_min) / (scores_max - scores_min + 1e-8))

      # scale labels from [-1,1] to [0,1]
      labels = (labels + 1) / 2

    return metrics.precision_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())

def recall(scores,labels,similarity_type, threshold):
    """Compute the recall for pair classification.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1. (Hamming & Euclidean), positive scores mean higher preference
        of being assigned to label +1 while negative scores mean higher preference of being assigned
        to label -1.  (Cosine) scores are setted to +1 or -1 based on the treshold imposed.
      labels: [n_examples] int.  Labels are either +1 or -1.
      similarity_type: the type of similarity used to compute the scores.
      threshold: the threshold used to binarize the scores.

    Returns:
      recall: the recall score.
    """

    if similarity_type == 'cosine':
      scores = torch.sign(scores + 1e-8 - threshold)
    
    else:
      scores_max = torch.max(scores)
      scores_min = torch.min(scores)

      # normalize scores to [0, 1] and add a small epislon for safety
      scores = torch.round((scores - scores_min) / (scores_max - scores_min + 1e-8))

      # scale labels from [-1,1] to [0,1]
      labels = (labels + 1) / 2

    return metrics.recall_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())

def accuracy(scores,labels,similarity_type, threshold):
    """Compute the accuracy for pair classification.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1. (Hamming & Euclidean), positive scores mean higher preference
        of being assigned to label +1 while negative scores mean higher preference of being assigned
        to label -1.  (Cosine) scores are setted to +1 or -1 based on the treshold imposed.
      labels: [n_examples] int.  Labels are either +1 or -1.
      similarity_type: the type of similarity used to compute the scores.
      threshold: the threshold used to binarize the scores.

    Returns:
      accuracy: the accuracy score.
    """

    if similarity_type == 'cosine':
      scores = torch.sign(scores + 1e-8 - threshold)
    
    else:
      scores_max = torch.max(scores)
      scores_min = torch.min(scores)

      # normalize scores to [0, 1] and add a small epislon for safety
      scores = torch.round((scores - scores_min) / (scores_max - scores_min + 1e-8))

      # scale labels from [-1,1] to [0,1]
      labels = (labels + 1) / 2

    return metrics.accuracy_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())

def confusion_matrix(scores,labels,similarity_type, threshold):
    """Compute the accuracy for pair classification.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1. (Hamming & Euclidean), positive scores mean higher preference
        of being assigned to label +1 while negative scores mean higher preference of being assigned
        to label -1. (Cosine) scores are setted to +1 or -1 based on the treshold imposed.
      labels: [n_examples] int.  Labels are either +1 or -1.
      similarity_type: the type of similarity used to compute the scores.
      threshold: the threshold used to binarize the scores.

    Returns:
      tn: true negative
      fp: false positive
      fn: false negative
      tp: true positive
    """

    if similarity_type == 'cosine':
      scores = torch.sign(scores + 1e-8 - threshold)
    
    else:
      scores_max = torch.max(scores)
      scores_min = torch.min(scores)

      # normalize scores to [0, 1] and add a small epislon for safety
      scores = torch.round((scores - scores_min) / (scores_max - scores_min + 1e-8))

      # scale labels from [-1,1] to [0,1]
      labels = (labels + 1) / 2

    tn, fp, fn, tp = metrics.confusion_matrix(labels.cpu().detach().numpy(), scores.cpu().detach().numpy()).ravel()

    return tn, fp, fn, tp



