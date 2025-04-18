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
from sklearn.model_selection import train_test_split

PACKERS = ['kkrunchy', 'mpress', 'obsidium', 'pecompact', 'pelock', 'petite', 'telock', 'themida', 'upx']

def file_analysis(filepath, db_path, model, log_filepath, config, normalization_mean, normalization_std):

        filename = filepath.split("/")[-1]
        print("filename: " + filename)

        if filepath.endswith(".xml"):

            graph_path = filepath
        
        else:

            print("Invalid file format!")
            exit(1)

        dataset = PackedGraphSimilarityPairs(db_path,graph_path,normalization_mean,normalization_std)

        # EVALUATION
        model.eval()
        start_time = time.time()

        print("\nEvaluating...")
        bar = IncrementalBar('Comparison', max = dataset.get_db_size())

        with torch.no_grad():
            similarities = defaultdict(lambda: dict())

            batch_size = config['evaluation']['batch_size']
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

        # Save name of analyzed files
        with open(log_filepath, mode="a") as f:

            f.write(filename + "\n")
        
        return similarities


for excluded_packer in PACKERS:

    for i in [2,5] + list(range(10, 100, 10)):
    
        print("TESTING AGAINST: " + excluded_packer + "\n")
        experiment_path = 'paper_evaluation/experiments_ph/againstml/no' + excluded_packer + '/0/' + str(i) + '/'
        model_path = experiment_path + 'packware_no' + excluded_packer + '.pt'
        log_dict_path = experiment_path

        test_path = 'paper_evaluation/graph_datasets/packware/integration/test/' + excluded_packer + '/'

        db_path = 'paper_evaluation/graph_datasets/packware/integration/train/finetuned/' + excluded_packer + '/'+ str(i) + '/'

        log_filepath = log_dict_path + "log.txt"
        similarities_filepath = log_dict_path + "similarities.pkl"

        # Set GPU
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')

        # import configuration
        config = get_default_config()

        db_dataset = TrainingPackedGraphSimilarityDataset(db_path,validation_size=config['data']['dataset_params']['validation_size'])
        # Extract normalization metrics from db
        normalization_mean, normalization_std, features_order = db_dataset.get_node_statistics()
        # Retrieve node and edge feature dimension
        node_feature_dim, edge_feature_dim = db_dataset.get_features_dim()

        # Build model from saved weights
        model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
        model.to(device)
        model.load_state_dict(torch.load(model_path))

        filenames = listdir(test_path)
        filenames = train_test_split(filenames, test_size=0.2139, random_state=518)[1]
        filenames.sort()

        # if the result filename exists, remove the files already analyzed from the list
        if os.path.isfile(log_filepath):
            with open(log_filepath) as f:
                datafile = f.readlines()
                for line in datafile:
                    print(line[:-1] + " already analyzed!")
                    if line[:-1] in filenames:
                        filenames.remove(line[:-1])
            
            with open(similarities_filepath, "rb") as f:
                unpickler = pickle.Unpickler(f)
                similarities_byfilename = unpickler.load()
        else:
            similarities_byfilename = {}

        for filename in filenames:

            filepath = test_path + filename
            similarities_byfilename[filename] = dict(file_analysis(filepath, db_path, model, log_filepath, config, normalization_mean, normalization_std))
            print("\n--------------------------------\n")

            with open(similarities_filepath, 'wb') as f:
                pickle.dump(similarities_byfilename, f)