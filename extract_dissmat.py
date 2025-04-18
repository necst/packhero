import numpy as np
import torch
from os import listdir
import pandas as pd
from collections import defaultdict

from tool_dependencies.configure import *
from graph_clustering.clustering_dataset import *
from tool_dependencies.utils import *
from tool_dependencies.evaluation import *


PACKERS = ['kkrunchy', 'mpress', 'obsidium', 'pecompact', 'pelock', 'petite', 'telock', 'themida', 'upx']

DB_PATH = 'paper_evaluation/graph_datasets/../'
EXPERIMENT_PATH = 'paper_evaluation/experiments_ph/againstsignature/accessible/'
MODEL_PATH = "paper_evaluation/experiments_ph/trainsetsize/100/packware_100.pt"

BATCH_SIZE = 256

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
filenames = [f for f in filenames if any(f.lower().startswith(packer) for packer in PACKERS)]

# Create a dissimilarity matrix for each packer in db

# extracted_packers = set([filename.split("_")[0] for filename in listdir(EXPERIMENT_PATH + 'dissmat_rgd1/')])
extracted_packers = []

for packer in set([filename.split("_")[0] for filename in filenames]):

    if packer in extracted_packers:
        print("Dissimilarity matrix for packer: ", packer, " already exists.\n")
        continue

    print("Processing packer: ", packer)

    similarities = defaultdict(lambda: np.array([]))
    current_file_num = 1

    filenames_by_packer = [filename for filename in filenames if filename.startswith(packer)]
    filenames_by_packer = sorted(filenames_by_packer)
    number_of_files = len(filenames_by_packer)

    for filename in filenames_by_packer:

        graph_path = DB_PATH + filename
        dataset = PackedGraphSimilarityPairs(DB_PATH,packer,graph_path,normalization_mean,normalization_std)

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

    # Fill the values of the upper triangular matrix
    for i in range(len(filenames_by_packer)):
        for j in range(i + 1, len(filenames_by_packer)):
            similarities[filenames_by_packer[i]] = np.append(similarities[filenames_by_packer[i]], similarities[filenames_by_packer[j]][i])

    cosine_similarity_matrix = pd.DataFrame.from_dict(similarities, orient='index', columns=filenames_by_packer)

    # Extract fixed threshold
    min_similarity = cosine_similarity_matrix.stack().mean() - cosine_similarity_matrix.stack().std()
    print(f"Adjusted mean similarity for {packer}: {min_similarity}")

    if os.path.exists(EXPERIMENT_PATH + "fixed_thresholds.pkl"):
        with open(EXPERIMENT_PATH + "fixed_thresholds.pkl", "rb") as f:
            thresholds = pickle.load(f)
        thresholds[packer] = min_similarity
    else:
        thresholds = {packer: min_similarity}

    print(thresholds)

    with open(EXPERIMENT_PATH + "fixed_thresholds.pkl", 'wb') as f:
        pickle.dump(thresholds, f)


    # Assuming 'cosine_similarity_matrix' is your precomputed cosine similarity matrix
    # Convert similarity to dissimilarity
    dissimilarity_matrix = 1 - cosine_similarity_matrix

    dissimilarity_metrix_copy = dissimilarity_matrix.copy()
    np.fill_diagonal(dissimilarity_metrix_copy.values, 0)  # Ensuring the diagonal is 0 (self-distance)

    # Save dissimilarity matrix to pickle file
    dissimilarity_metrix_copy.to_pickle(EXPERIMENT_PATH + 'dissmat/' + packer + '_dissmat.pkl')

    print("Done processing packer: ", packer)
    print("--------------------------------------------------")
    print()