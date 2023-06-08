import os
import json
import copy

import torch.multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import sparse

import torch
from torch.utils.data import Dataset, DataLoader

from utils import *

class UrbanGraph(Dataset):

    def __init__(self, config, city):

        self.config = config
        self.city = city

        self.region_attributes = self.load_region_attributes()
        self.regions = self.load_regions()
        self.num_region = len(self.regions)
        self.adjacency = self.load_adjacency()
        self.distance = self.load_distance()
        self.roadnet = self.load_roadnet()
        self.rail = self.load_rail()
        self.POI_similarity = self.load_POI_similarity()

        self.OD = self.load_od()

    def load_regions(self):
        return [x for x in range(self.region_attributes.shape[0])]

    def load_region_attributes(self):
        path = self.config["data_path"] + self.city
        region_attributes = np.load(path + "/region_attributes.npy")
        return region_attributes

    def load_adjacency(self):
        path = self.config["data_path"] + self.city
        adjacency = np.load(path + "/adjacency.npy")
        return adjacency

    def load_distance(self):
        path = self.config["data_path"] + self.city
        distance = np.load(path + "/distance.npy")
        return distance

    def load_roadnet(self):
        return None

    def load_rail(self):
        return None

    def load_POI_similarity(self):
        return None

    def load_od(self):
        path = self.config["data_path"] + self.city
        od = np.load(path + "/od.npy")
        return od

    def sample_random_walk(self):
        node_seq = []
        feat_seq = []
        dis_seq = []
        edge_seq = []
        seq = []

        init_node = np.random.choice(self.regions)

        node_seq.append(init_node)
        feat_seq.append(self.region_attributes[init_node])
        for i in range(self.config["len_random_walk"] -1):
            p = self.OD[node_seq[-1]][self.OD[node_seq[-1]].nonzero()]
            p = p / p.sum()
            next_node = np.random.choice(self.OD[node_seq[-1]].nonzero()[0], p = p)
            feat = self.region_attributes[next_node]
            flow_between = self.OD[node_seq[-1], next_node]
            dis_between = self.distance[node_seq[-1], next_node]
            node_seq.append(next_node)
            feat_seq.append(feat)
            edge_seq.append(flow_between)
            dis_seq.append(dis_between)

        # node_seq = node_seq[1:]
        # node_seq = np.eye(self.num_region)[node_seq]
        feat_seq = np.array(feat_seq[1:])
        edge_seq = np.array(edge_seq).reshape([-1, 1])
        dis_seq = np.array(dis_seq).reshape([-1, 1])
        # seq = np.concatenate([node_seq, edge_seq, dis_seq], axis=1) # node_seq, feat_seq, edge_seq, dis_seq
        seq = edge_seq
        return seq


class Graphs(Dataset):

    def __init__(self, config):

        self.config = config

        self.cities = self.get_graphs_name(config["data_path"])

        self.training_cities = self.load_training_cities()
        self.target_city = self.config["target_city"]
        self.graphs = {}

        self.add_graphs()


    def __getitem__(self, index):
        pass

    def __len__(self):
        return None

    def get_graphs_name(self, path):
        names = list(os.walk(path))[0][1]
        return names

    def load_training_cities(self):
        training_cities = copy.deepcopy(self.cities)
        training_cities.remove(self.config["target_city"])
        return training_cities

    def add_a_graph(self, city):
        graph = UrbanGraph(self.config, city)
        self.graphs[city] = graph

    def add_graphs(self):
        for city in self.cities:
            self.add_a_graph(city)

    def sample_random_walk_from_a_training_graph(self):
        graph_selected = np.random.choice(self.training_cities)
        return self.graphs[graph_selected].sample_random_walk()

    def sample_random_walk_from_target_city_graph(self):
        return self.graphs[self.config["target_city"]].sample_random_walk()

    def sample_target_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.sample_random_walk_from_target_city_graph())
        batch = np.stack(batch)

        # mp.set_start_method("spawn")
        # batch = zip(*pool.map(self.sample_random_walk_from_target_city_graph))
        # batch = list(pool.apply_async(self.sample_random_walk_from_target_city_graph))
        # print(batch)
        # exit(0)
        # for i in range(batch_size):
        #     results.append(pool.apply_async(self.sample_random_walk_from_target_city_graph))
        # pool.close()
        # pool.join()
        # for res in results:
        #     batch.append(res.get())
        # batch = np.stack(batch)

        return batch

    def sample_training_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.sample_random_walk_from_a_training_graph())
        batch = np.stack(batch)
        return batch
