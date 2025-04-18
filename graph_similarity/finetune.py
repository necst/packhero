from evaluation import compute_similarity, auc
from loss import pairwise_loss
from utils import *
from configure import *
from finetune_dataset import *
import numpy as np
import torch.nn as nn
import collections
import time
import random
import math
import gc
import os
from time import sleep


# Proportion of the number of train and validation pairs to mantain for each packer
PAIRS_PROP = 1 # % of the pairs are used, increase or decrease this value to change the number of pairs used

early_stopping = "train" # "train" or "val"
epsilon = 1e-2 # Minimum improvement to consider as improvement

def main(DATASET_PATH, MODEL_PATH, PRETRAINED_MODEL, packers = []):

    # Set GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    batch_pool = []

    print("\ndevice: ", device.type)

    print("\nHYPERPARAMETERS\n")

    # Print configure
    config = get_default_config()
    for (k, v) in config.items():
        print("%s = %s" % (k, v))

    print("\nFINE-TUNING STARTED\n")

    # Set random seeds
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    # Build dataset
    if config['data']['problem'] == 'CG_similarity':
      dataset_params = config['data']['dataset_params']
      validation_size = dataset_params['validation_size']
      dataset = FineTuningPackedGraphSimilarityDataset(DATASET_PATH, packers, validation_size=validation_size, pairs_prop=PAIRS_PROP)
    else:
      raise ValueError('Unknown problem type: %s' % config['data']['problem'])

    # Retrieve node and edge feature dimension
    node_feature_dim, edge_feature_dim = dataset.get_features_dim()

    # Build model
    model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
    model.to(device)
    model.load_state_dict(torch.load(PRETRAINED_MODEL))

    print("MODEL in RAM")

    # Metrics for each Packer
    accumulated_metrics = collections.defaultdict(list)

    # Set number of graphs in batch
    training_n_graphs_in_batch = config['training']['batch_size']
    # There are 2 graphs for each pair, so we double the number of graphs in batch
    training_n_graphs_in_batch *= 2

    # Set number of iterations to number of batches
    config['training']['n_training_steps'] = math.floor(dataset.get_train_pairs_size() / config['training']['batch_size'])

    print("Number of pairs: ", dataset.get_train_pairs_size())
    print("Number of training steps: ", config['training']['n_training_steps'])
    print()

    # Set early stopping
    best_val_score = float('-inf')  # Adjust depending on whether lower or higher is better
    best_model_statedict = None
    no_improve_epochs = 0
    patience = 5  # Number of epochs to wait for improvement before stopping

    t_start_iter = time.time()
    t_start_epoch = time.time()

    # EPOCHS

    for epoch in range(1,config['training']['epochs'] + 1):

        gc.collect()
        torch.cuda.empty_cache()

        print("EPOCH %d" % (epoch))

        i_iter = 0
        sum_sim_diff = 0
        model.train(mode=True)

        # Train with all data for each epoch
        if epoch == 1:
            data_iter = dataset.pairs(config['training']['batch_size'], type = 'train')
        else:
            data_iter = batch_pool

        for elem in data_iter:

            if epoch == 1:
                batch_pool.append(elem)

            node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(elem)
            
            labels = labels.to(device)
            graph_vectors = model(node_features.to(device), edge_features.to(device), from_idx.to(device), to_idx.to(device), graph_idx.to(device), training_n_graphs_in_batch)

            x, y = reshape_and_split_tensor(graph_vectors, 2)
            loss = pairwise_loss(x, y, labels,
                                    loss_type=config['training']['loss'],
                                    margin=config['training']['margin'])

            is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
            is_neg = 1 - is_pos
            n_pos = torch.sum(is_pos)
            n_neg = torch.sum(is_neg)
            sim = compute_similarity(config, x, y)
            sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)
            sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8)

            graph_vec_scale = torch.mean(graph_vectors ** 2)

            if config['training']['graph_vec_regularizer_weight'] > 0:
                loss += (config['training']['graph_vec_regularizer_weight'] *
                        0.5 * graph_vec_scale)
            
            optimizer.zero_grad()
            grad_tensor = torch.ones_like(loss, device=loss.device)
            time1 = time.time()
            loss.backward(grad_tensor)
            time2 = time.time()
            #print("Time to backward: ", time2 - time1)
            nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_value'])
            optimizer.step()

            sim_diff = sim_pos - sim_neg
            accumulated_metrics['loss'].append(loss)
            accumulated_metrics['sim_pos'].append(sim_pos)
            accumulated_metrics['sim_neg'].append(sim_neg)
            accumulated_metrics['sim_diff'].append(sim_diff)

            sum_sim_diff += sim_diff

            # EVALUATION

            if (i_iter + 1) % config['training']['print_after'] == 0 or i_iter == config['training']['n_training_steps'] - 1:
                metrics_to_print = {
                    k: torch.mean(v[0]) for k, v in accumulated_metrics.items()}
                info_str = ', '.join(
                    ['%s %.4f' % (k, v) for k, v in metrics_to_print.items()])
                # clear the accumulated metrics
                accumulated_metrics.clear()

                # validation set
                # we evaluate the validation set only if it exists (when we do the final training we use the entire set)
                if config['data']['dataset_params']['validation_size'] > 0:
                    if ((i_iter + 1) // config['training']['print_after'] % config['training']['eval_after'] == 0) or i_iter == config['training']['n_training_steps'] - 1:
                        model.eval()
                        with torch.no_grad():
                            similarities = torch.tensor([])
                            y_val = torch.tensor([])
                            for batch in dataset.pairs(config['evaluation']['batch_size'], type = 'val'):
                                node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
                                labels = labels.to(device)
                                eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                                to_idx.to(device),
                                                graph_idx.to(device), config['evaluation']['batch_size'] * 2)

                                x, y = reshape_and_split_tensor(eval_pairs, 2)
                                similarity = compute_similarity(config, x, y)

                                similarities = torch.concatenate((similarities, similarity))
                                y_val = torch.concatenate((y_val, labels))

                            # AUC metric computation
                            pair_auc = auc(similarities, y_val)

                            eval_metrics = {
                                'pair_auc': pair_auc}
                            info_str += ', ' + ', '.join(
                                ['%s %.4f' % ('val/' + k, v) for k, v in eval_metrics.items()])
                        model.train(mode=True)
                print('epoch %d, iter %d, %s, time %.2fs' % (epoch,
                    i_iter + 1, info_str, time.time() - t_start_iter))
                t_start_iter = time.time()

            i_iter += 1

        print("epoch %d training time: %.2f s" % (epoch, time.time() - t_start_epoch))

        t_start_epoch = time.time()
        # Check if the current model is better than the previous best
        if early_stopping == "train":
            early_metric = sum_sim_diff / i_iter
        elif early_stopping == "val":
            early_metric = eval_metrics['pair_auc']

        if early_metric > best_val_score + epsilon:  # Use > for metrics where higher is better
            best_val_score = early_metric
            best_model_statedict = model.state_dict()
            no_improve_epochs = 0

            print(f'Epoch {epoch}: New best early_metric: {early_metric:.4f}')
        else:
            no_improve_epochs += 1

        # Check for early stopping
        if no_improve_epochs >= patience:
            print(f'No improvement in early_metric for {patience} consecutive epochs. Stopping training.')
            break

    # Save trained model to use it after
    torch.save(best_model_statedict, MODEL_PATH)

if __name__ == '__main__':

    DATASETS_PATH = "../paper_evaluation/graph_datasets/packware/integration/train/finetuned"

    for i in [2,5]:
        # iterate over all the directories in the datasets folder
        for dataset in os.listdir(DATASETS_PATH):
            DATASET_PATH = DATASETS_PATH + "/" + dataset + "/" + str(i) + "/"
            PRETRAINED_MODEL = '../paper_evaluation/experiments_ph/againstml/no' + dataset + '/0/100/packware_no' + dataset + '.pt'
            MODEL_PATH = '../paper_evaluation/experiments_ph/againstml/no' + dataset + '/' + str(i) + '/packware_' + dataset + '.pt'
            if dataset == "themida":
                dataset = "themida-v2"
            main(DATASET_PATH, MODEL_PATH, PRETRAINED_MODEL, packers = [dataset])
            sleep(5)