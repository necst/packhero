import pickle
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split


EXPERIMENT_PATH = f"paper_evaluation/experiments_ph/trainsetsize/100/"
RESULTS_PATH = EXPERIMENT_PATH + "identification_files/"
SIMILARITIES_PATH = EXPERIMENT_PATH + "similarities_100.pkl"
CLUSTERING_PATH = EXPERIMENT_PATH + "clustering.pkl"
THRESHOLDS_PATH = EXPERIMENT_PATH + "fixed_thresholds.pkl"
THRESHOLDS_INTRACLUSTER_PATH = EXPERIMENT_PATH + "fixed_thresholds_intracluster.pkl"

with open(SIMILARITIES_PATH, "rb") as f:
    similarities = pickle.load(f)

with open(CLUSTERING_PATH, 'rb') as f:
    clusters_info = pickle.load(f)

with open(THRESHOLDS_PATH, 'rb') as f:
    thresholds = pickle.load(f)

with open(THRESHOLDS_INTRACLUSTER_PATH, 'rb') as f:
    thresholds_intracluster = pickle.load(f)

# if clustering and noclustering results exists drop it
if os.path.exists(RESULTS_PATH + 'noclustering.txt'):
    os.remove(RESULTS_PATH + 'noclustering.txt')

if os.path.exists(RESULTS_PATH + 'clustering.txt'):
    os.remove(RESULTS_PATH + 'clustering.txt')

print(f"Extracting identification...")

# similarities = {filename: {packer: {filename_target: similarity}}}

#test_samples = train_test_split(os.listdir(f"paper_evaluation/graph_datasets/packware/integration/test/{external_packer}/"), test_size=0.2139, random_state=518)[1]

for filename in tqdm(similarities):

    print(filename)

    # if filename not in test_samples:
    #     continue

    num_inferences = 0

    results_mean = {}

    no_clustering_res = open(RESULTS_PATH + 'noclustering.txt', mode="a")
    # Compute proportion of matches for each packer and choose max if it is greater than threshold_matches (the graph has matched with at least threshold_matches*100% of the packer's graphs)
    scores = defaultdict(lambda: 0)
    for packer in similarities[filename]:
        similarities_array = np.fromiter(similarities[filename][packer].values(), dtype=float)
        for similarity in similarities_array:
            num_inferences += 1
            if similarity >= thresholds[packer]:
                scores[packer] += 1
        scores[packer] = scores[packer]/len(similarities_array) # Proportion of matches

    max_score = max(scores.values())
    identified_packer = max(scores, key=scores.get) + " (percentage of matches with the identified packer: " + str(round(max_score*100, 2)) + "%)"

    if identified_packer.split(" ")[0].lower() == filename.split("_")[0]:
        no_clustering_res.write("filename: " + filename + "\n")
        no_clustering_res.write(identified_packer.split(" ")[0].lower() + " - MATCH - num_inferences: " + str(num_inferences) + "\n")
    else:
        no_clustering_res.write("filename: " + filename + "\n")
        no_clustering_res.write(filename.split("_")[0].lower() + " - MISMATCH (" + identified_packer.split(" ")[0] + ") - num_inferences: " + str(num_inferences) + "\n")

    no_clustering_res.close()

    # Clustering part

    clustering_res = open(RESULTS_PATH + 'clustering.txt', mode="a")

    medoids = [clusters_info['filenames'][index] for index in clusters_info['medoids']]
    # Delete from medoids the packers not in similarities
    medoids = [medoid for medoid in medoids if medoid.split('_')[0] in similarities[next(iter(similarities))].keys()]

    # Clustering part (select only valuable clusters with similarity > 0)
    closer_clusters_medoids = [medoid for medoid in medoids if similarities[filename][medoid.split('_')[0]][medoid] > 0]
    num_inferences_clustering = len(medoids)

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
        for filename_target in filenames:
            if similarities[filename][filename_target.split("_")[0]][filename_target] >= thresholds_intracluster[clusters_info['cluster_labels'][clusters_info['filenames'].index(filename_target)]]:
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

    if identified_packer.split(" ")[0].lower() == filename.split("_")[0]:
        clustering_res.write("filename: " + filename + "\n")
        clustering_res.write(identified_packer.split(" ")[0].lower() + " - MATCH - num_inferences: " + str(num_inferences_clustering) + "\n")
    else:
        
        clustering_res.write("filename: " + filename + "\n")
        clustering_res.write(filename.split("_")[0].lower() + " - MISMATCH (" + identified_packer.split(" ")[0] + ") - num_inferences: " + str(num_inferences_clustering) + "\n")
        
    clustering_res.close()
