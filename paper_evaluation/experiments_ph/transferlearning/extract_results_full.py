#!/usr/bin/python3
import os
from os import walk
import csv
from collections import defaultdict
from sklearn import metrics
import pandas as pd
import numpy as np

EXPERIMENT_PATH = f"./"
RESULTS_PATH = EXPERIMENT_PATH + "identification_files/"

files = [filenames for (dirpath, dirnames, filenames) in walk(RESULTS_PATH)][0]

# if res.csv already exists, delete it to rewrite it
if os.path.exists(EXPERIMENT_PATH + 'res.csv'):
    os.remove(EXPERIMENT_PATH + 'res.csv')

csv_file = open(EXPERIMENT_PATH + 'res.csv', 'a')
csv_file_writer = csv.writer(csv_file)
csv_file_writer.writerow(['approach', 'packer', 'version', 'config', 'samples_num', 'identification_rate'])

count = 0
for file in files:

    match = defaultdict(lambda: list())
    num_instances_map = defaultdict(lambda: list())

    approach = file.split(".")[0]

    print(f"approach: {approach}")

    f = open(RESULTS_PATH + file, "r")
    lines = f.readlines()

    for line in lines:
        if line.startswith("filename"):
            filename = line.split(": ")[1]
            version = filename.split("-")[0].split("_")[-1]
            print(version)
            config = filename.split("-")[1]
            print(config)
        else:
            packer = line.split(" - ")[0].lower()
            result = line.split(" - ")[1]
            num_instances = int(line.split(" - ")[-1][16:])

            if result == "MATCH":
                match[packer+"-"+version+"-"+config].append(True)
            else:
                match[packer+"-"+version+"-"+config].append(False)
    

    eval_metrics = {
        'samples_num': [len(match[packer]) for packer in match],
        'identification_rate': [round(sum(match[packer])/len(match[packer]), 2) for packer in match]
    }


    indexes = [packer for packer in match]

    metr = pd.DataFrame.from_dict(eval_metrics).set_index(pd.Index(indexes))

    for index, row in metr.iterrows():
        csv_file_writer.writerow([approach, index.split("-")[0], index.split("-")[1], index.split("-")[2], row['samples_num'], row['identification_rate']])

    mean = metr.mean().round(2)
    metr.loc['------'] = '------'
    metr.loc['macro-avg'] = mean
    metr.loc['macro-avg', 'samples_num'] = int(metr.loc['macro-avg', 'samples_num'])
    print(metr.to_markdown())

    print()