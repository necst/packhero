import torch
import time
import pickle
from os import listdir
from collections import defaultdict
from progress.bar import IncrementalBar

from tool_dependencies.extract_custom_gcg import *
from tool_dependencies.configure import *
from tool_dependencies.tool_dataset import *
from tool_dependencies.utils import *
from tool_dependencies.evaluation import *

from collections import Counter

DB_PATH = "paper_evaluation/graph_datasets/packgenome/RGD-complete/"
DATASET_PATH = "paper_evaluation/graph_datasets/packgenome/LPD/"
MODEL_PATH = "paper_evaluation/experiments_ph/trainsetsize/100/packware_100.pt"
EXPERIMENT_PATH = 'paper_evaluation/experiments_ph/againstsignature/accessible/'
RESULTS_PATH = EXPERIMENT_PATH + "identification_files/"
CLUSTERING_PATH = EXPERIMENT_PATH + "clustering.pkl"
THRESHOLDS_INTRACLUSTER_PATH = "paper_evaluation/experiments_ph/trainsetsize/100/fixed_thresholds_intracluster.pkl"

if os.path.exists(RESULTS_PATH + 'clustering.txt'):
    os.remove(RESULTS_PATH + 'clustering.txt')

with open(CLUSTERING_PATH, 'rb') as f:
    clusters_info = pickle.load(f)

with open(THRESHOLDS_INTRACLUSTER_PATH, 'rb') as f:
    thresholds_intracluster = pickle.load(f)

clustering_res = open(RESULTS_PATH + 'clustering.txt', mode="a")

# Set GPU
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

def file_analysis(filepath):

    filename = filepath.split("/")[-1]
    print("filename: " + filename)

    if filepath.endswith(".xml"):

        graph_path = filepath
    
    else:

        print("Invalid file format!")
        exit(1)

    medoids = [clusters_info['filenames'][index] for index in clusters_info['medoids']]
    num_inferences_clustering = len(medoids)

    dataset_medoids = PackedGraphSimilarityPairs(DB_PATH,graph_path,normalization_mean,normalization_std, included_files=medoids)

    # EVALUATION
    model.eval()
    start_time = time.time()

    print("\nEvaluating Medoids...")
    bar = IncrementalBar('Comparison', max = dataset_medoids.get_db_size())

    with torch.no_grad():

        similarities = defaultdict(lambda: dict())

        batch_size = 8
        files_analyzed = 0

        start_time = time.time()
        for batch_graphs, batch_packers, batch_files in dataset_medoids.pairs(batch_size):
            node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(batch_graphs)
            eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                to_idx.to(device),
                                graph_idx.to(device), batch_size * 2)
            
            x, y = reshape_and_split_tensor(eval_pairs, 2)
            similarity = compute_similarity(config, x, y)

            if files_analyzed + batch_size > dataset_medoids.get_db_size():
                        
                for i in range(dataset_medoids.get_db_size() - files_analyzed):
                        similarities[batch_packers[i]][batch_files[i]] = similarity[i].item()
                bar.next(dataset_medoids.get_db_size() - files_analyzed)

            else:

                for i in range(batch_size):
                    similarities[batch_packers[i]][batch_files[i]] = similarity[i].item()

                bar.next(batch_size)

                files_analyzed += batch_size

        bar.finish()

        closer_clusters_medoids = [medoid for medoid in medoids if similarities[medoid.split("_")[0]][medoid] > 0]
        if len(closer_clusters_medoids) == 0:
            identified_packer = "Unknown"
        else:       
            closer_clusters_labels = [clusters_info['cluster_labels'][clusters_info['filenames'].index(closer_cluster_medoid)] for closer_cluster_medoid in closer_clusters_medoids]
            filenames = [clusters_info['filenames'][index] for index in range(len(clusters_info['filenames'])) if clusters_info['cluster_labels'][index] in closer_clusters_labels]
            num_inferences_clustering += len(filenames)
            # presence of packer in the subset of the closer clusters (counter of packer in the subset of the closer clusters)
            num_packer_in_subset = Counter([f.split("_")[0] for f in filenames])
            max_num_packer_in_subset = max(num_packer_in_subset.values())

            scores = defaultdict(lambda: 0)

            similarities = defaultdict(lambda: dict())

            dataset = PackedGraphSimilarityPairs(DB_PATH,graph_path,normalization_mean,normalization_std, included_files=filenames)

            start_time = time.time()

            print("\nEvaluating Intra-Clustering...")
            bar = IncrementalBar('Comparison', max = dataset.get_db_size())

            batch_size = 1
            files_analyzed = 0

            start_time = time.time()
            for batch_graphs, batch_packers, batch_files in dataset.pairs(batch_size):
                node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(batch_graphs)
                eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                    to_idx.to(device),
                                    graph_idx.to(device), batch_size * 2)
                
                x, y = reshape_and_split_tensor(eval_pairs, 2)
                similarity = compute_similarity(config, x, y)

                if files_analyzed + batch_size > dataset.get_db_size():
                            
                    for i in range(dataset.get_db_size() - files_analyzed):
                            similarities[batch_packers[i]][batch_files[i]] = similarity[i].item()
                    bar.next(dataset.get_db_size() - files_analyzed)

                else:

                    for i in range(batch_size):
                        similarities[batch_packers[i]][batch_files[i]] = similarity[i].item()

                    bar.next(batch_size)

                    files_analyzed += batch_size

            
            bar.finish()

            print("\nEvaluation time: ", time.time() - start_time)

            for filename_target in filenames:
                if similarities[filename_target.split("_")[0]][filename_target] >= thresholds_intracluster[clusters_info['cluster_labels'][clusters_info['filenames'].index(filename_target)]]:
                    scores[filename_target.split("_")[0]] += 1
                else:
                    scores[filename_target.split("_")[0]] += 0

            for packer in scores:
                scores[packer] = scores[packer]/max_num_packer_in_subset # Proportion of matches with the most frequent packer in the subset of the closer clusters

            max_score = max(scores.values())

            if max_score == 0:
                identified_packer = max(num_packer_in_subset, key=num_packer_in_subset.get) + " (percentage of matches with the identified packer: 0%)" # If no similarity is above the threshold, the identified packer is the most frequent packer in the subset of the closer clusters (percentage of matches with the identified packer: 0%
            else:
                identified_packer = max(scores, key=scores.get) + " (percentage of matches with the identified packer: " + str(round(max_score*100, 2)) + "%)"

            print(scores)
            print(identified_packer)

        if identified_packer.split(" ")[0].lower().startswith(filename.split("_")[0]):
            clustering_res.write("filename: " + filename + "\n")
            clustering_res.write(identified_packer.split(" ")[0].lower() + " - MATCH - num_inferences: " + str(num_inferences_clustering) + "\n")
        else:
            clustering_res.write("filename: " + filename + "\n")
            clustering_res.write(filename.split("_")[0].lower() + " - MISMATCH (" + identified_packer.split(" ")[0] + ") - num_inferences: " + str(num_inferences_clustering) + "\n")

filenames = listdir(DATASET_PATH)
filenames.sort()
# remove files not starting with PACKERS = ['kkrunchy','mpress','obsidium','pecompact', 'pelock', 'petite', 'telock', 'themida', 'upx']
PACKERS = ['kkrunchy','mpress','obsidium','pecompact', 'petite', 'themida', 'upx']
filenames = [f for f in filenames if any(f.startswith(packer) for packer in PACKERS)]

for filename in filenames:

    filepath = DATASET_PATH + filename
    file_analysis(filepath)
    print("\n--------------------------------\n")

clustering_res.close()