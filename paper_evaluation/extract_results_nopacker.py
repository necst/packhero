#!/usr/bin/python3
import os
from os import walk
import csv
from collections import defaultdict
from sklearn import metrics
import pandas as pd
import numpy as np

PACKERS = ['telock', 'themida', 'upx']

for external_packer in PACKERS:

    for i in [2,5] + list(range(10, 110, 10)):

        EXPERIMENT_PATH = "experiments_ph/againstml/no" + external_packer + "/0/" + str(i) + "/"
        RESULTS_PATH = EXPERIMENT_PATH + "identification_files/"

        files = os.listdir(RESULTS_PATH)

        # if res.csv already exists, delete it to rewrite it
        if os.path.exists(EXPERIMENT_PATH + 'res.csv'):
            os.remove(EXPERIMENT_PATH + 'res.csv')


        csv_file = open(EXPERIMENT_PATH + 'res.csv', 'a')
        csv_file_writer = csv.writer(csv_file)
        csv_file_writer.writerow(['approach', 'packer', 'samples_num', 'recall', 'avg_num_instances'])

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
                    continue
                packer = line.split(" - ")[0].lower()
                result = line.split(" - ")[1]
                num_instances = int(line.split(" - ")[-1][16:])

                if result == "MATCH":
                    match[packer].append(packer)
                else:
                    match[packer].append(result.split("(")[1][:-1])

                

                num_instances_map[packer].append(num_instances)
            

            y_test = np.array([])
            pred = np.array([])
            unknown_rate = np.array([])

            for packer in match:
                num_samples = len(match[packer])
                y_test = np.concatenate((y_test, np.array([packer for i in range(num_samples)])))
                pred = np.concatenate((pred, np.array(match[packer])))
                unknown_rate = np.append(unknown_rate, match[packer].count('Unknown') / num_samples)

            truen = []
            falsep = []
            falsen = []
            truep = []

            flattened_mul_cm = metrics.multilabel_confusion_matrix(y_test, pred, labels = sorted(list(set(y_test)))).ravel()

            for i in range(0,len(flattened_mul_cm), 4):
                truen.append(flattened_mul_cm[i])
                falsep.append(flattened_mul_cm[i+1])
                falsen.append(flattened_mul_cm[i+2])
                truep.append(flattened_mul_cm[i+3])

            eval_metrics = {
                'samples_num': [len(match[packer]) for packer in sorted(list(set(y_test)))],
                'recall': [round((truep[i] / (truep[i] + falsen[i])), 2) if (truep[i] + falsen[i]) > 0 else 0 for i in range(len(list(set(y_test))))],
                'avg_num_instances': [round(np.mean(num_instances_map[packer]), 2) for packer in sorted(list(set(y_test)))]
            }


            indexes = [packer for packer in match]

            metr = pd.DataFrame.from_dict(eval_metrics).set_index(pd.Index(indexes))

            for index, row in metr.iterrows():
                csv_file_writer.writerow([approach, index, row['samples_num'], row['recall'], row['avg_num_instances']])

            mean = metr.mean().round(2)
            print(metr.to_markdown())

            print()
        

        

    

