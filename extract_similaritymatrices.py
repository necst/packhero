import numpy as np
import torch
from os import listdir
import pandas as pd
from collections import defaultdict

from tool_dependencies.configure import *
from graph_clustering.clustering_dataset import *
from tool_dependencies.utils import *
from tool_dependencies.evaluation import *
from sklearn.model_selection import train_test_split

DB_PATH = 'paper_evaluation/graph_datasets/packware/entire/test-small/'
EXPERIMENT_PATH = 'paper_evaluation/experiments_ph/similaritymatrices/'
MODEL_PATH = "paper_evaluation/experiments_ph/trainsetsize/100/packware_100.pt"

BATCH_SIZE = 512

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# import configuration
config = get_default_config()

db_dataset = TrainingPackedGraphSimilarityDataset(DB_PATH,validation_size=config['data']['dataset_params']['validation_size'])
# Extract normalization metrics from db
normalization_mean, normalization_std, features_order = db_dataset.get_node_statistics()

# Retrieve node and edge feature dimension
node_feature_dim, edge_feature_dim = db_dataset.get_features_dim()

# Build model from saved weights
model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))

filenames = listdir(DB_PATH)
filenames = sorted(filenames)

similarities = defaultdict(lambda: np.array([]))
current_file_num = 1

number_of_files = len(filenames)

for filename in filenames:

    graph_path = DB_PATH + filename
    dataset = PackedGraphSimilarityPairs(DB_PATH,None,graph_path,normalization_mean,normalization_std)

    print(f'Processing file {current_file_num} of {number_of_files}...')

    batch_size = dataset.get_db_size()

    with torch.no_grad():

        for batch_graphs, batch_files in dataset.pairs(batch_size):
            node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(batch_graphs)
            eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                            to_idx.to(device),
                            graph_idx.to(device), number_of_files * 2)

            x, y = reshape_and_split_tensor(eval_pairs, 2)
            similarities_batch = compute_similarity(config, x, y)

            for i in range(len(batch_files)):

                current_file = batch_files[i]
                similarity = similarities_batch[i].item()
                
                if similarity > 1:
                    similarities[filename] = np.append(similarities[filename], 1)
                else:
                    similarities[filename] = np.append(similarities[filename], similarity)

        
        similarities[filename] = np.append(similarities[filename], 1) # Add the similarity of the file with itself

    current_file_num += 1

print(similarities)
# Fill the values of the upper triangular matrix
for i in range(len(filenames)):
    for j in range(i + 1, len(filenames)):
        similarities[filenames[i]] = np.append(similarities[filenames[i]], similarities[filenames[j]][i])

cosine_similarity_matrix = pd.DataFrame.from_dict(similarities, orient='index', columns=filenames)
# save the cosine similarity matrix with pickle
cosine_similarity_matrix.to_pickle(EXPERIMENT_PATH + 'cosine_similarity_matrix.pkl')
