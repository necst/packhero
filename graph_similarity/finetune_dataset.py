import abc
import random
import collections
import os

import numpy as np
import networkx as nx

from itertools import combinations_with_replacement, groupby
from sklearn.model_selection import train_test_split

"""A general Interface"""

RANDOM_STATE = 42


class GraphSimilarityDataset(object):
    """Base class for all the graph similarity learning datasets.
  This class defines some common interfaces a graph similarity dataset can have,
  in particular the functions that creates iterators over pairs.
  """

    @abc.abstractmethod
    def pairs(self, batch_size, type='train'):
        """Create an iterator over pairs.
    Args:
      batch_size: int, number of pairs in a batch.
      type: str, 'train' or 'validation'.
    Yields:
      graphs: a `GraphData` instance.  The batch of pairs put together.  Each
        pair has 2 graphs (x, y).  The batch contains `batch_size` number of
        pairs, hence `2*batch_size` many graphs.
      labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
    """
        pass


"""Packed PEs Call Graph Task"""


# Graph Manipulation Functions
def permute_graph_nodes(g):
    """Permute node ordering of a graph, returns a new permuted graph."""
    n = g.number_of_nodes()
    perm = np.random.permutation(n)
    
    permuted_graph = g.copy()
    perm_dict = {i: perm[i] for i in range(n)}
    permuted_graph = nx.relabel_nodes(permuted_graph, perm_dict)

    return permuted_graph


class FineTuningPackedGraphSimilarityDataset(GraphSimilarityDataset):
    """Dataset for graph matching problems."""
    def __init__(self, dataset_path, packers, validation_size = 0.1, permute=True, pairs_prop = 1):
      """Constructor.

      Args:
        dataset_path: path where we can find the train dataset.
        validation_size: proportion of the dataset to use for validation.
        permute: if True (default), permute node orderings in addition to
          changing edges; if False, the node orderings across a pairs of
          graphs will be the same, useful for visualization.
      """
      self._permute = permute
      self._dataset_path = dataset_path
      self._packers = packers

      # Take files from path and split them into train and test files
      files = os.listdir(self._dataset_path)

      if validation_size > 0:
        self._train_files, self._validation_files = train_test_split(files, test_size=validation_size, stratify=[file.split("_")[0] for file in files] ,random_state=RANDOM_STATE)
        self._validation_pairs = self.__gen_stratified_pairs__(self._validation_files, pairs_prop=pairs_prop)
      else:
        self._train_files = files
        self._validation_files = []

      self._train_pairs = self.__gen_stratified_pairs__(self._train_files, self._packers, pairs_prop=pairs_prop)
      
      # Parameters for normalization (train set) (to be saved in the dataset to use later for test data)
      node_features = []
      for file in self._train_files:
        g = nx.convert_node_labels_to_integers(nx.graphml.read_graphml(self._dataset_path + file))
        node_features.append(np.array([[*g.nodes.data()[i].values()] for i in range(g.number_of_nodes())], dtype=np.float32))
      
      node_features = np.concatenate(node_features, dtype=np.float32)

      self._node_mean = np.mean(node_features, axis=0)
      self._node_std = np.std(node_features, axis=0)
      self._features_order = [*g.nodes.data()[0].keys()]

      self._edge_feature_dim = 4

    def __normalize_node__(self, arr):
      """Normalize node features with pre-calculated statistics.

      Args:
        arr: array of values.

      Returns:
        array of normalized values.
      """
      features = arr.copy()

      for i in range(len(features)):
        if self._node_std[i] != 0 and self._node_mean[i] != 0:
          features[i] = (features[i] - self._node_mean[i]) / self._node_std[i]
      
      return features

    def __gen_stratified_pairs__(self, files, packers, neg_pos_prop = 1, pairs_prop = 1):
      """Generate pairs stratified for positive and negative classes (similar and dissimilar).

      Args:
        files: a list files.
        neg_pos_prop: proportion of negative pairs respect to positive pairs (default = 1).
        pairs_prop: proportion of pairs (default = 1)

      Returns:
        pairs: a list of pairs of files.
      """

      pairs = []
      files_sorted = np.array(sorted(files))

      # Split files by packer
      keyfunct = lambda x: x.split("_")[0]
      grouper = groupby(files_sorted, keyfunct)

      files_by_packer = []
      for _, group in grouper:
        shuffled = list(group)
        random.Random(RANDOM_STATE).shuffle(shuffled)
        files_by_packer.append(shuffled)
      random.Random(RANDOM_STATE).shuffle(files_by_packer)
      
      # Generate pairs
      for i in range(len(files_by_packer)):

        if files_by_packer[i][0].split("_")[0] not in packers:
           continue
        
        pos_pairs = list(combinations_with_replacement(files_by_packer[i], 2))
        # in order to have a limited number of positive pairs (and, consequently, a limited number of
        # negative pairs, we can select only a portion of the combinations_with_replacement previously obtained)
        pos_pairs = pos_pairs[:int((len(pos_pairs)) * pairs_prop)]
        
        pairs += pos_pairs

        j = (i + 1) % len(files_by_packer) 
        l = k = 0

        for _ in range(len(pos_pairs) * neg_pos_prop):
          pairs.append((files_by_packer[i][l], files_by_packer[j][k]))
  
          l = (l + 1) % len(files_by_packer[i])
          j = (j + 1) % len(files_by_packer)
          
          if j == i:
            j = (j + 1) % len(files_by_packer)
          
          k = (k + 1) % len(files_by_packer[j])

      random.Random(RANDOM_STATE).shuffle(pairs)
        
      return pairs

    def __pack_batch__(self, graphs):
        """Pack a batch of graphs into a single `GraphData` instance.
    Args:
      graphs: a list of networkx graphs.
    Returns:
      graph_data: a `GraphData` instance, with node and edge indices properly
        shifted.
    """
        Graphs = []
        for graph in graphs:
            for inergraph in graph:
                Graphs.append(inergraph)
        graphs = Graphs
        from_idx = []
        to_idx = []
        graph_idx = []
        node_features = []

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            edges = np.array(g.edges(), dtype=np.int32)
            # shift the node indices for the edges
            if len(edges) > 0:
                from_idx.append(edges[:, 0] + n_total_nodes)
                to_idx.append(edges[:, 1] + n_total_nodes)
            else:
                from_idx.append(np.zeros(0, dtype=np.int32))
                to_idx.append(np.zeros(0, dtype=np.int32))
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)
            node_features.append(np.array([self.__normalize_node__([*g.nodes.data()[i].values()]) for i in range(n_nodes)], dtype=np.float32))

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        GraphData = collections.namedtuple('GraphData', [
            'from_idx',
            'to_idx',
            'node_features',
            'edge_features',
            'graph_idx',
            'n_graphs'])

        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            node_features=np.concatenate(node_features, dtype=np.float32),
            # this task only cares about structures and node features, 
            # the graphs have no edge features. setting higher dimension 
            # of ones to confirm code functioning with high dimensional features.
            edge_features=np.ones((n_total_edges, self._edge_feature_dim), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )

    def get_features_dim(self):
      """return dimension of node and edge features.

      Returns:
        dimension of node and edge features
      """

      return len(self._features_order), self._edge_feature_dim

    def get_node_statistics(self):
      """return statistics for each node feature.

      Returns:
        statistics of node features
      """

      return self._node_mean, self._node_std, self._features_order
    
    def get_train_pairs_size(self):
       """return number of train pairs.

      Returns:
        number of train pairs
      """
       
       return len(self._train_pairs)
    
    def pairs(self, batch_size, type='train'):
          
      batch_graphs = []
      batch_labels = []

      if type == 'train':
        pairs = self._train_pairs
      else:
        pairs = self._validation_pairs
      
      for pair in pairs[1:]:

        g1 = nx.convert_node_labels_to_integers(nx.graphml.read_graphml(self._dataset_path + pair[0]))
        g2 = nx.convert_node_labels_to_integers(nx.graphml.read_graphml(self._dataset_path + pair[1]))

        if self._permute:
          g1 = permute_graph_nodes(g1)
          g2 = permute_graph_nodes(g2)

        packer1 = pair[0].split("_")[0]
        packer2 = pair[1].split("_")[0]

        batch_graphs.append((g1, g2))
        batch_labels.append(1 if (packer1 == packer2) else -1)
        del g1
        del g2

        if len(batch_graphs) == batch_size:
          yield self.__pack_batch__(batch_graphs), np.array(batch_labels)
          batch_graphs.clear()
          batch_labels.clear()

      # If last batch is not full, fill it with first pairs.
      if len(batch_graphs) > 0:
        for pair in pairs:

          g1 = nx.convert_node_labels_to_integers(nx.graphml.read_graphml(self._dataset_path + pair[0]))
          g2 = nx.convert_node_labels_to_integers(nx.graphml.read_graphml(self._dataset_path + pair[1]))

          if self._permute:
            g1 = permute_graph_nodes(g1)
            g2 = permute_graph_nodes(g2)

          packer1 = pair[0].split("_")[0]
          packer2 = pair[1].split("_")[0]

          batch_graphs.append((g1, g2))
          batch_labels.append(1 if (packer1 == packer2) else -1)
          del g1
          del g2

          if len(batch_graphs) == batch_size:
            yield self.__pack_batch__(batch_graphs), np.array(batch_labels)
            batch_graphs.clear()
            batch_labels.clear()
            break