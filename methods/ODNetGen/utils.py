import os
import random
import sys
import time
import json
import copy

import dgl

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn.functional as fct

import torch.multiprocessing as mp


class MyLogger(object):
    def __init__(self, config):
        super(MyLogger, self).__init__()
        self.config = config

        self.generator_path = config["exp_path"] + "models/generators/" + config["exp"] + "_" + config["target_city"] + "_G.pkl"
        self.discriminator_path = config["exp_path"] + "models/discriminators/" + config["exp"] + "_" + config["target_city"] + "_D.pkl"

        self.metrics_path = config["exp_path"] + "logs/" + config["exp"] + "_" + config["target_city"] + ".json"

        self.metrics = {}

        self.best_training_loss = float("inf")
        self.best_jsd_net = None


    def log_metrics(self, metric_name, value):
        if metric_name not in self.metrics.keys():
            self.metrics[metric_name] = [value]
        else:
            self.metrics[metric_name].append(value)

    def check_training_loss_decent(self, training_metric):
        if training_metric < self.best_training_loss:
            self.best_training_loss = training_metric
            # print(" * Reach a better training stage...\n")
            return True
        else:
            return False

    def log_model_weights(self, path, model_G):
        save_path = path
        torch.save(model_G.state_dict(), save_path)

    def log_generated_ODNetwork(self, network):
        save_path = self.config["exp_path"] + "generation/" + self.config["exp"] + "_" + self.config["target_city"] + ".npy"
        np.save(save_path, network)

    def reload_metrics(self):
        with open(self.metrics_path, "r") as f:
            data = json.load(f)
        self.metrics = data["metrics"]
        self.best_training_loss = np.min(self.metrics["RMSE"])

    def log_best_net(self, net):
        self.best_jsd_net = net

    def metrics_to_file(self):
        save_path = self.config["exp_path"] + "logs/" + self.config["exp"] + "_" + self.config["target_city"] + ".json"

        tmp_config = copy.deepcopy(self.config)

        del tmp_config["g"]
        del tmp_config["device"]

        log = {
                "exp_config" : tmp_config,
                "metrics" : self.metrics
                }
        with open(save_path, "w") as f:
            json.dump(log, f, indent=4, sort_keys=True)

def gpu_info(mem_need = 10000):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')

    mem_idx = [2, 6, 10, 14, 18, 22, 26, 30]
    mem_bus = [x for x in range(8)]
    mem_list = []
    for idx, info in enumerate(gpu_status):
        if idx in mem_idx:
            mem_list.append(11019 - int(info.split('/')[0].split('M')[0].strip()))
    idx = np.array(mem_bus).reshape([-1, 1])
    mem = np.array(mem_list).reshape([-1, 1])
    id_mem = np.concatenate((idx, mem), axis=1)
    GPU_available = id_mem[id_mem[:,1] >= mem_need][:,0]

    if len(GPU_available) != 0:
        return GPU_available
    else:
        return None

def narrow_setup(interval = 0.5, mem_need = 2000):
    GPU_available = gpu_info()
    i = 0
    while GPU_available is None:  # set waiting condition
        GPU_available = gpu_info(mem_need)
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        sys.stdout.write('\r' + ' ' + symbol)
        sys.stdout.flush()
        # time.sleep(interval)
        i += 1
    GPU_selected = random.choice(GPU_available)
    return GPU_selected

# graph topology of NYC and graph construction
def build_graph_from_matrix(adjm, device):
    # get edge nodes' tuples [(src, dst)]
    dst, src = adjm.nonzero()
    # get edge weights
    d = adjm[adjm.nonzero()]
    # create a graph
    g = dgl.graph(adjm.nonzero(), device = device)
    g = g.add_self_loop()
    return g

def skew_sigmoid_torch(v):
    v = 1 / ( 1 + torch.exp(-50*v) )
    return v

def skew_sigmoid_numpy(v):
    v = 1 / ( 1 + np.exp(-50*v) )
    return v

# Calculates the gradient penalty loss for WGAN GP
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.FloatTensor(real_samples.shape[0]).fill_(1.0).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def one_hot_2_integer(one_hot):
    idx = torch.argmax(one_hot)
    one_hot[:idx] = one_hot[:idx] + 1
    one_hot[idx] = one_hot[idx] - 1
    return one_hot.sum()

def RMSE(preds, target):
    return np.sqrt(np.mean((preds - target) **2))

def prob_vec(data):
    prob_list = []
    prob_list.append(np.sum(data < 0))
    prob_list.append(np.sum(data == 0))
    prob_list.append(np.sum((data > 0) & (data <= 1)))
    prob_list.append(np.sum((data > 1) & (data <= 2)))
    prob_list.append(np.sum((data > 2) & (data <= 3)))
    prob_list.append(np.sum((data > 3) & (data <= 4)))
    prob_list.append(np.sum((data > 4) & (data <= 5)))
    prob_list.append(np.sum((data > 5) & (data <= 6)))
    prob_list.append(np.sum((data > 6) & (data <= 7)))
    prob_list.append(np.sum((data > 7) & (data <= 8)))
    prob_list.append(np.sum((data > 8) & (data <= 9)))
    prob_list.append(np.sum((data > 9) & (data <= 10)))
    prob_list.append(np.sum((data > 10) & (data <= 15)))
    prob_list.append(np.sum((data > 15) & (data <= 20)))
    prob_list.append(np.sum((data > 20) & (data <= 30)))
    prob_list.append(np.sum((data > 30) & (data <= 50)))
    prob_list.append(np.sum((data > 50) & (data <= 75)))
    prob_list.append(np.sum((data > 75) & (data <= 100)))
    prob_list.append(np.sum((data > 100) & (data <= 150)))
    prob_list.append(np.sum((data > 150) & (data <= 200)))
    prob_list.append(np.sum((data > 200) & (data <= 300)))
    prob_list.append(np.sum((data > 300) & (data <= 400)))
    prob_list.append(np.sum((data > 400) & (data <= 500)))
    prob_list.append(np.sum((data > 500) & (data <= 600)))
    prob_list.append(np.sum((data > 600) & (data <= 700)))
    prob_list.append(np.sum((data > 700) & (data <= 800)))
    prob_list.append(np.sum((data > 800) & (data <= 900)))
    prob_list.append(np.sum((data > 900) & (data <= 1000)))
    prob_list.append(np.sum((data > 1000) & (data <= 1200)))
    prob_list.append(np.sum((data > 1200) & (data <= 1500)))
    prob_list.append(np.sum((data > 1500) & (data <= 1800)))
    prob_list.append(np.sum((data > 1800) & (data <= 2000)))
    prob_list.append(np.sum((data > 2000) & (data <= 3000)))
    prob_list.append(np.sum((data > 3000) & (data <= 4000)))
    prob_list.append(np.sum((data > 4000) & (data <= 5000)))
    prob_list.append(np.sum((data > 5000) & (data <= 6000)))
    prob_list.append(np.sum((data > 6000) & (data <= 7000)))
    prob_list.append(np.sum((data > 7000) & (data <= 8000)))
    prob_list.append(np.sum((data > 8000) & (data <= 9000)))
    prob_list.append(np.sum((data > 9000) & (data <= 10000)))
    prob_list.append(np.sum(data > 10000))
    prob = np.array(prob_list)
    prob = prob / prob.sum()
    return prob
