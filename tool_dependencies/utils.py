import collections
from tool_dependencies.train_dataset import TrainingPackedGraphSimilarityDataset
from tool_dependencies.graphembeddingnetwork import GraphEmbeddingNet, GraphEncoder, GraphAggregator
from tool_dependencies.graphmatchingnetwork import GraphMatchingNet
import copy
import torch
import pickle

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs'])


def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split


def build_model(config, node_feature_dim, edge_feature_dim):
    """Create model for training and evaluation.

    Args:
      config: a dictionary of configs, like the one created by the
        `get_default_config` function.
      node_feature_dim: int, dimensionality of node features.
      edge_feature_dim: int, dimensionality of edge features.

    Returns:
      tensors: a (potentially nested) name => tensor dict.
      placeholders: a (potentially nested) name => tensor dict.
      AE_model: a GraphEmbeddingNet or GraphMatchingNet instance.

    Raises:
      ValueError: if the specified model or training settings are not supported.
    """
    config['encoder']['node_feature_dim'] = node_feature_dim
    config['encoder']['edge_feature_dim'] = edge_feature_dim

    encoder = GraphEncoder(**config['encoder'])
    aggregator = GraphAggregator(**config['aggregator'])
    if config['model_type'] == 'embedding':
        model = GraphEmbeddingNet(
            encoder, aggregator, **config['graph_embedding_net'])
    elif config['model_type'] == 'matching':
        model = GraphMatchingNet(
            encoder, aggregator, **config['graph_matching_net'])
    else:
        raise ValueError('Unknown model type: %s' % config['model_type'])

    optimizer = torch.optim.Adam((model.parameters()),
                                 lr=config['training']['learning_rate'], weight_decay=1e-5)

    return model, optimizer


def build_datasets(config, dataset_path, pairs_prop = 1):
    """Build the training and evaluation datasets."""
    config = copy.deepcopy(config)

    if config['data']['problem'] == 'CG_similarity':
      dataset_params = config['data']['dataset_params']
      validation_size = dataset_params['validation_size']
      dataset = TrainingPackedGraphSimilarityDataset(dataset_path, validation_size=validation_size, pairs_prop=pairs_prop)
    else:
      raise ValueError('Unknown problem type: %s' % config['data']['problem'])
    return dataset


def get_graph(graph):
    
    node_features = torch.from_numpy(graph.node_features)
    edge_features = torch.from_numpy(graph.edge_features)
    from_idx = torch.from_numpy(graph.from_idx).long()
    to_idx = torch.from_numpy(graph.to_idx).long()
    graph_idx = torch.from_numpy(graph.graph_idx).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
