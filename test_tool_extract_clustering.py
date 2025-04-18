import pandas as pd
from os import listdir
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import numpy as np
from collections import defaultdict, Counter
import pickle

EXPERIMENT_PATH = f'paper_evaluation/experiments_ph/againstsignature/accessible/'
DISSIMILARITY_MATRICES_PATH = EXPERIMENT_PATH + 'dissmat/'

max_cluster_size = 1  # Parameter to control the maximum size of clusters to merge
dissimilarity_matrices = {}

for filename in sorted(listdir(DISSIMILARITY_MATRICES_PATH)):
    dissimilarity_matrices[filename.split("_")[0]] = pd.read_pickle(DISSIMILARITY_MATRICES_PATH + filename)

Z_dict = {}

for packer in dissimilarity_matrices:
    dm = ssd.squareform(dissimilarity_matrices[packer])
    Z = linkage(dm, 'single')
    Z_dict[packer] = Z

max_k = 10
hierarchical_silhouette_scores = []
best_ks = {}

for packer in Z_dict:
    Z = Z_dict[packer]
    hierarchical_silhouette_s = []
    for k in range(2, max_k):
        hierarchical_l = fcluster(Z, k, criterion='maxclust')
        if len(set(hierarchical_l)) == 1:
            hierarchical_silhouette_s.append(0)
            continue
        if len(set(hierarchical_l)) == dissimilarity_matrices[packer].shape[0]:
            hierarchical_silhouette_s.append(0)
            continue
        score = silhouette_score(dissimilarity_matrices[packer], hierarchical_l, metric='precomputed')
        hierarchical_silhouette_s.append(score)
    best_ks[packer] = np.argmax(hierarchical_silhouette_s) + 2

hierarchical_labels_by_packer = {}
packer_labels = np.array([], dtype='str')

for packer in Z_dict:
    Z = Z_dict[packer]
    k = best_ks[packer]
    clustering = fcluster(Z, k, criterion='maxclust')
    hierarchical_labels_by_packer[packer] = clustering

    packer_labels = np.concatenate((packer_labels, np.full(len(clustering), packer)))

done = False
mod = 0
old_mod = 0

while not done:

    for packer in hierarchical_labels_by_packer:

        for i in range(1, max(hierarchical_labels_by_packer[packer]) + 1):
            cluster_indexes = [j for j, x in enumerate(hierarchical_labels_by_packer[packer]) if x == i]
            
            # Check if the cluster size is within the specified limit
            if 1 <= len(cluster_indexes) <= max_cluster_size:
                min_cluster = 0
                min_distance = float('inf')  # Initialize with infinity

                for sample_index in cluster_indexes:
                    for j in range(1, max(hierarchical_labels_by_packer[packer]) + 1):
                        if i != j:  # Skip the same cluster
                            other_cluster_indexes = [k for k, x in enumerate(hierarchical_labels_by_packer[packer]) if x == j]
                            # Scale sample index and other cluster indexes to the range of the dissimilarity matrix
                            distance = dissimilarity_matrices[packer].iloc[sample_index, other_cluster_indexes].min()
                            if distance < min_distance:
                                min_distance = distance
                                min_cluster = j
                    
                    mod += 1
                    hierarchical_labels_by_packer[packer][sample_index] = min_cluster

                # Adjust cluster numbers for subsequent clusters
                for k in range(len(hierarchical_labels_by_packer[packer])):
                    if hierarchical_labels_by_packer[packer][k] > i:
                        hierarchical_labels_by_packer[packer][k] -= 1
    
    if mod == old_mod:
        done = True
    else:
        old_mod = mod

def find_medoids(dissimilarity_matrices, cluster_labels_by_packer):
    medoids = defaultdict(lambda: [])
    for packer in hierarchical_labels_by_packer:
        for cluster_id in np.unique(cluster_labels_by_packer[packer]):
            # Extract the indices of points within this cluster
            indices = np.where(cluster_labels_by_packer[packer] == cluster_id)[0]
            # Extract the submatrix of dissimilarities corresponding to these points
            cluster_dissimilarity_matrix = dissimilarity_matrices[packer].iloc[indices, indices]

            # Find the medoid (point with the smallest total distance to all others in the cluster)
            medoid_index = indices[np.argmin(cluster_dissimilarity_matrix.sum(axis=1))]
            medoids[packer].append(medoid_index)
    
    return medoids

hierarchical_medoids_by_packer = find_medoids(dissimilarity_matrices, hierarchical_labels_by_packer)

# Create a single array of clusterings labels and a single array of medoids
hierarchical_labels = np.array([], dtype='int')
hierarchical_medoids = np.array([], dtype='int')

local_to_global_index = {}

for packer in hierarchical_labels_by_packer:
    if len(local_to_global_index) != 0:
        max_index = 0
        for p in local_to_global_index:
            if max(local_to_global_index[p]) > max_index:
                max_index = max(local_to_global_index[p])
        max_index += 1
        local_to_global_index[packer] = [i + max_index for i, x in enumerate(hierarchical_labels_by_packer[packer])]
    else:
        local_to_global_index[packer] = [i for i, x in enumerate(hierarchical_labels_by_packer[packer])]

for packer in hierarchical_labels_by_packer:
    if len(hierarchical_labels) == 0:
        hierarchical_labels = np.concatenate((hierarchical_labels, hierarchical_labels_by_packer[packer]))
    else:
        hierarchical_labels = np.concatenate((hierarchical_labels, hierarchical_labels_by_packer[packer] + hierarchical_labels.max()))

    # Adjust medoid indexes that are relative to packer index to be relative to the entire dataset
    for index in hierarchical_medoids_by_packer[packer]:
        hierarchical_medoids = np.append(hierarchical_medoids, local_to_global_index[packer][index])


# Mapping each packer to its cluster
cluster_mapping = {}
for cluster_id, packer in zip(hierarchical_labels, packer_labels):
    if cluster_id not in cluster_mapping:
        cluster_mapping[cluster_id] = []
    cluster_mapping[cluster_id].append(packer)

# Counting the frequency of packers in each cluster
cluster_packer_frequency = {cluster: Counter(packers) for cluster, packers in cluster_mapping.items()}

# Convert to DataFrame
df_cluster_distribution = pd.DataFrame.from_dict(cluster_packer_frequency, orient='index').fillna(0)
df_cluster_distribution = df_cluster_distribution.astype(int)  # Convert counts to integers
df_cluster_distribution.sort_index(inplace=True)  # Sort by cluster id

print(df_cluster_distribution.transpose())

filenames = np.array([], dtype='str')
for packer in hierarchical_labels_by_packer:
    filenames = np.concatenate((filenames, dissimilarity_matrices[packer].columns))

clusters = {
    'filenames': filenames.tolist(),
    'cluster_labels': hierarchical_labels,
    'medoids': hierarchical_medoids
}

# Save the threshold for each cluster, i.e., the mean similarity of the similarities (converted from the dissimilarities) where dissimilarity_matrix = 1 - cosine_similarity_matrix
# retrieve the packer matrix from the cluster dict
cluster_thresholds = {}
for medoid in hierarchical_medoids:
    packer = filenames[medoid].split("_")[0]
    cluster_id = hierarchical_labels[filenames.tolist().index(filenames[medoid])]
    for cluster_intrapacker_id in np.unique(hierarchical_labels_by_packer[packer]):
        cluster_indexes = np.where(hierarchical_labels_by_packer[packer] == cluster_intrapacker_id)[0]
        cluster_matrix = 1 - dissimilarity_matrices[packer].iloc[cluster_indexes, cluster_indexes]
        cluster_thresholds[cluster_id] = cluster_matrix.stack().mean() - cluster_matrix.stack().std()

print(cluster_thresholds)

with open(EXPERIMENT_PATH + "fixed_thresholds_intracluster.pkl", 'wb') as f:
    pickle.dump(cluster_thresholds, f)

with open(EXPERIMENT_PATH + 'clustering.pkl', 'wb') as f:
    pickle.dump(clusters, f)