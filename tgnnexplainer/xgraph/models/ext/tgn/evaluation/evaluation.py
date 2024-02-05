import math
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def eval_simulate_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # assert negative_edge_sampler.seed is not None
    # negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    val_acc = []
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in tqdm(range(num_test_batch), total=num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]
            labels = data.labels[s_idx: e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size) # useless

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                negative_samples, timestamps_batch,
                                                                edge_idxs_batch, n_neighbors)

            # pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            # true_label = np.concatenate([np.ones(size), np.zeros(size)])
            pred_score = (pos_prob).cpu().numpy()
            pred_label = (pred_score >= 0.5).flatten().astype(float)
            true_label = np.array(labels)
            # import ipdb; ipdb.set_trace()

            val_ap.append(average_precision_score(true_label, pred_score))
            # val_auc.append(roc_auc_score(true_label, pred_score))
            val_acc.append( (pred_label==true_label).mean() )

    # return np.mean(val_ap), np.mean(val_auc), np.mean(val_acc)
    return np.mean(val_acc), np.mean(val_ap)


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    val_acc = []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                negative_samples, timestamps_batch,
                                                                edge_idxs_batch, n_neighbors)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            pred_label = (pred_score >= 0.5).flatten().astype(float)
            # import ipdb; ipdb.set_trace()

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
            val_acc.append( (pred_label==true_label).mean() )

    return np.mean(val_ap), np.mean(val_auc), np.mean(val_acc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx: e_idx]

            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                                                                                                     destinations_batch,
                                                                                                                                                                     destinations_batch,
                                                                                                                                                                     timestamps_batch,
                                                                                                                                                                     edge_idxs_batch,
                                                                                                                                                                     n_neighbors)
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
