#!/usr/bin/env python3

from os import walk
from os import path
import sys

from extract_custom_gcg import *
from tqdm import tqdm

# Set path to the packed PE dataset
DATASET_PATH_PACKED = "../../dataset-packed-pe/not-packed"
GENERATED_DATASET_PATH = "../paper_evaluation/graph_datasets/dataset-packed-pe/not-packed/"
dataset = 'not-packed' # True for packgenome dataset
PACKED_FILES = [(dirpath, filenames) for (dirpath, dirnames, filenames) in walk(DATASET_PATH_PACKED)][1:]
if PACKED_FILES == []:
    PACKED_FILES = [(dirpath, filenames) for (dirpath, dirnames, filenames) in walk(DATASET_PATH_PACKED)]

# List of files that have been discarded because they corresponds to a graph with more than 500 nodes
# Leave only .exe packed files
if dataset == 'packgenome-rgd':
    for i in range(len(PACKED_FILES)):
        packer = PACKED_FILES[i][0].split("/")[7]
        PACKED_FILES[i] = (PACKED_FILES[i][0], [file for file in PACKED_FILES[i][1] if file.endswith("_packed.exe")])

if dataset == 'packgenome-rgd1':
    for i in range(len(PACKED_FILES)):
        PACKED_FILES[i] = (PACKED_FILES[i][0], [file for file in PACKED_FILES[i][1] if file.endswith(".exe")])

discarded_list = []

for i in range(len(PACKED_FILES)):
    if PACKED_FILES[i][1] == []:
        continue

    if dataset == 'packed_pe_outliers' or dataset == 'packware' or 'packgenome-rgd' or 'packgenome-lpd' or 'not-packed':
        packer_name = ""
        version_name = ""
    else:
        packer_name = PACKED_FILES[i][0].split("/")[-1] + '_'
        version_name = ""

    print(packer_name, version_name)

    print("Generating...")

    for j in tqdm(range(len(PACKED_FILES[i][1]))):

        packed_filepath = PACKED_FILES[i][0]
        packed_filename = PACKED_FILES[i][1][j]

        if not path.exists(GENERATED_DATASET_PATH + packer_name + version_name + packed_filename[:-4] + ".xml") or "--rigenerate" in sys.argv:
            
            if "--gvzplots" in sys.argv:

                G, dot_graph = extract_gcg_with_plot(packed_filepath + '/' + packed_filename)
                if G == None:
                    discarded_list.append(packer_name + version_name + packed_filename)
                else:
                    save_graph_pdf(dot_graph, GENERATED_DATASET_PATH + packer_name + version_name + packed_filename)

            else:

                G = extract_gcg(packed_filepath + '/' + packed_filename)
                if G == None:
                    discarded_list.append(packer_name + version_name + packed_filename)
                

            if "--graphml" in sys.argv and G != None:
                save_graph_networkx(G, GENERATED_DATASET_PATH + packer_name + version_name + packed_filename)

print("\nDiscarded files: ", discarded_list)