import abc
import collections
import os

import numpy as np
import networkx as nx

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

class PackedGraphSimilarityPairs(GraphSimilarityDataset):
    """Dataset pf pairs between the given graphs and the db graphs."""
    def __init__(self, db_path, packer, graph_path, normalization_mean, normalization_std):
      """Constructor.

      Args:
        db_path: path where we can find the dataset to match with.
        graph_path: path where we can find the graph to match.
        normalization_mean: mean to use to normalize node features of input graph.
        normalization_mean: std to use to normalize node features of input graph.
      """
      self._db_path = db_path
      self._packer = packer
      self._graph_path = graph_path
      self._normalization_mean = normalization_mean
      self._normalization_std = normalization_std

      # Take files from paths
      self._db_files = [file for file in os.listdir(self._db_path) if packer == None or file.startswith(self._packer)]
      self._db_files = sorted(self._db_files)
      # Select files until the graph_path file is found
      self._db_files = self._db_files[:self._db_files.index(self._graph_path.split('/')[-1])]
      
      '''
      random.seed(42)
      self._db_files = random.sample(self._db_files, int(len(self._db_files)*1//10))
      '''

    def __normalize_node__(self, arr):
      """Normalize node features with pre-calculated statistics.

      Args:
        arr: array of values.

      Returns:
        array of normalized values.
      """
      features = arr.copy()

      for i in range(len(features)):
        if self._normalization_std[i] != 0 and self._normalization_mean[i] != 0:
          features[i] = (features[i] - self._normalization_mean[i]) / self._normalization_std[i]
      
      return features
    
    def get_db_size(self):
      """Return db size.
      """
      return len(self._db_files)
    
    def pairs(self, batch_size):
        
        batch_graphs = []
        batch_files = []
       
        for db_file in self._db_files:

          g1 = nx.convert_node_labels_to_integers(nx.graphml.read_graphml(self._graph_path))
          g2 = nx.convert_node_labels_to_integers(nx.graphml.read_graphml(self._db_path + db_file))

          batch_graphs.append((g1, g2))
          batch_files.append(db_file)

          if len(batch_graphs) == batch_size:
            yield self._pack_batch(batch_graphs), batch_files
            batch_graphs = []
            batch_files = []

        # If last batch is not full, fill it with first pairs.
        if len(batch_graphs) > 0:
          for db_file in self._db_files:

            g1 = nx.convert_node_labels_to_integers(nx.graphml.read_graphml(self._graph_path))
            g2 = nx.convert_node_labels_to_integers(nx.graphml.read_graphml(self._db_path + db_file))

            batch_graphs.append((g1, g2))
            batch_files.append(db_file)

            if len(batch_graphs) == batch_size:
              yield self._pack_batch(batch_graphs), batch_files
              batch_graphs = []
              batch_files = []
              break

    def _pack_batch(self, graphs):
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
            edge_features=np.ones((n_total_edges, 4), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )