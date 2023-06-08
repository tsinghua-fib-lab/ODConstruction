import time

import torch
import torch.nn.functional as fct
import torch.multiprocessing as mp


def sample_generated_batch(adjacency, logp, region_attributes, distance, config):
    start_time = time.time()

    params = [(adjacency, logp, region_attributes, distance, config)] * config["batch_size"]
    batch = []

    pool = mp.Pool(config["batch_size"])
    batch = pool.map(sample_one_random_walk, params)
    # batch = torch.stack(batch)
    print("walk sample time cost:", time.time() - start_time)
    exit(0)

def sample_one_random_walk(params):
    adjacency, trans_prob, node_feats, distance, config = params

    node_seq = []
    feat_seq = []
    edge_seq = []
    dis_seq = []

    init_node = torch.randint(low = 0, high = adjacency.size(0), size = (1,), device = config["device"])[0]
    init_node = fct.one_hot(init_node, num_classes = adjacency.size(0))

    node_seq.append(init_node)
    feat_seq.append(one_hot_to_nodefeat(init_node, node_feats, config))

    for i in range(config["len_random_walk"] -1):

        next_node = sample_a_neighbor(trans_prob, node_seq[-1], config)
        feat = one_hot_to_nodefeat(next_node, node_feats, config)
        flow_between = adjacency[torch.argmax(node_seq[-1]), torch.argmax(next_node)]
        dis_between = distance[torch.argmax(node_seq[-1]), torch.argmax(next_node)]

        node_seq.append(next_node)
        feat_seq.append(feat)
        edge_seq.append(flow_between)
        dis_seq.append(dis_between)

    # node_seq = torch.stack(node_seq[1:])
    # feat_seq = torch.stack(feat_seq[1:])
    edge_seq = torch.stack(edge_seq).view([-1, 1])
    # dis_seq = torch.stack(dis_seq).view([-1, 1])
    # seq = torch.cat((node_seq, edge_seq, dis_seq), dim=1) # node_seq, feat_seq, edge_seq, dis_seq
    seq = edge_seq
    print("one.")
    return seq

def one_hot_to_nodefeat(node, all_feats, config):
    node_idx = torch.argmax(node)

    feats = all_feats[:,:-config["noise_dim"]]
    node = node.unsqueeze(dim=1)[node_idx].repeat(feats.size(1))
    feat = feats[node_idx] * node
    return feat

def sample_a_neighbor(trans_prob, node, config):
    o_idx = torch.argmax(node)

    node = node.unsqueeze(dim=1)[o_idx].repeat(node.size(0))
    neighbors = node * trans_prob[o_idx]
    next_node = fct.gumbel_softmax(neighbors, tau = config["temperature"], hard=True)
    return next_node
