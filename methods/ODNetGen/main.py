import json
import os
import time
import datetime
import gc

import numpy as np

from math import *

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import jensenshannon

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import elu, relu, tanh, sigmoid
from torchviz import make_dot
import multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

from data_loader import *
from model import *
from mp_utils import *

import setproctitle



def main(config = None):

    writer = SummaryWriter(log_dir="exp/runs/", flush_secs=10)

    # data
    print("     loading data ...     \n")
    dataset = Graphs(config)

    # use target graph to train, test the mimic ability
    target_graph = dataset.graphs[dataset.target_city]

    node_feats = torch.FloatTensor(target_graph.region_attributes).to(config["device"])
    distance = torch.FloatTensor(target_graph.distance).to(config["device"])
    noise = torch.rand(target_graph.num_region, config["noise_dim"]).to(config["device"])
    node_feats = torch.cat((node_feats, noise), dim = 1)

    config["num_in"] = node_feats.size(1)
    config["g"] = build_graph_from_matrix(target_graph.adjacency, config["device"])
    config["random_walk_dim"] =  1 # target_graph.region_attributes.shape[1] +

    # model
    generator = Generator(config).to(config["device"])
    discriminator = Discriminator(config).to(config["device"])
    if config["model"] == "init":
        print("     Init models...    \n")
    elif config["model"] == "load":
        generator.load_state_dict(torch.load(logger.generator_path, map_location=config["device"]))
        discriminator.load_state_dict(torch.load(logger.discriminator_path, map_location=config["device"]))
        logger.reload_metrics()
        print("     Load models...    \n")


    # optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config["generator_lr"])
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config["discriminator_lr"])

    # training
    for epoch in range(config["EPOCH"]):

        start_time = time.time()

        if epoch < config["warm_epochs"]:
            config["n_G"] = config["n_G_warm"]
            config["n_D"] = config["n_D_warm"]
        else:
            config["n_G"] = config["n_G_stable"]
            config["n_D"] = config["n_D_stable"]

        noise = torch.rand(target_graph.num_region, config["noise_dim"]).to(config["device"])
        node_feats = torch.cat((node_feats[:,:target_graph.region_attributes.shape[1]], noise), dim = 1)


        # Train the generator every n_critic iterations
        if epoch % config["n_D"] == 0:
            """
            Train discriminator.
            """

            optimizer_D.zero_grad()

            # # generate graph
            # generator.generate_OD_net(node_feats)

            # generate fake samples by random walk
            with torch.no_grad():
                # fake_batch = generator.sample_generated_batch(node_feats, distance, config["batch_size"]).to(config["device"])

                generator.generate_OD_net(node_feats, distance)
                fake_batch = sample_generated_batch(generator.adjacency, generator.logp, node_feats, distance, config)
                print("sample_generated_batch complete.")

            # generate real samples by random walk
            real_batch = dataset.sample_target_batch(config["batch_size"])
            real_batch = torch.FloatTensor(real_batch).to(config["device"])

            # discriminate
            loss_D = - torch.mean(discriminator(real_batch)) + \
                       torch.mean(discriminator(fake_batch)) + \
                       config["weight_gp"] * compute_gradient_penalty(discriminator, real_batch, fake_batch, config["device"])

            if epoch != 0:
                loss_D.backward()
                optimizer_D.step()

            # clean some memory
            loss_D = loss_D.item()
            del fake_batch, real_batch
            gc.collect()

        # Train the generator every n_critic iterations
        if epoch % config["n_G"] == 0:
            """
            Train generator.
            """

            optimizer_G.zero_grad()

            # # generate graph
            # generator.generate_OD_net(node_feats)

            # generate fake samples by random walk
            fake_batch = generator.sample_generated_batch(node_feats, distance, config["batch_size"]).to(config["device"])

            # Adverserial loss
            loss_G = -torch.mean(discriminator(fake_batch))

            if epoch != 0:
                loss_G.backward()
                optimizer_G.step()

            # clean some memory
            loss_G = loss_G.item()
            del fake_batch
            gc.collect()

        generate_OD_net_numpy = generator.adjacency.detach().cpu().numpy()

        JSD = jensenshannon(prob_vec(generate_OD_net_numpy.reshape([-1])), prob_vec(target_graph.OD.reshape([-1])))
        rmse = RMSE(generate_OD_net_numpy, target_graph.OD)

        print("| EPOCH.", epoch, " | loss_D = {:.5f} | loss_G = {:.5f} | JSD = {:.4f} | RMSE = {:.2f} | time = {:.1f}".format(loss_D, loss_G, JSD, rmse, time.time() - start_time))

        writer.add_scalar("Train/loss_D", loss_D, epoch)
        writer.add_scalar("Train/loss_G", loss_G, epoch)
        writer.add_scalar("Train/W_distance", -loss_D, epoch)
        writer.add_scalar("DEBUG/G_parameter", generator.g_constructor.gnn.gat_layers[0].fc.weight[0,0], epoch)
        writer.add_scalars("DEBUG/gravtiy_parameter", \
                          {"G" : generator.g_constructor.G,
                           "lambda1" : generator.g_constructor.lambda1,
                           "lambda2" : generator.g_constructor.lambda2,
                           "lambda3" : generator.g_constructor.lambda3,
                           "beta" : generator.g_constructor.beta
                           }, \
                          epoch)
        writer.add_scalar("DEBUG/check_OD_generated_max", generate_OD_net_numpy.max(), epoch)
        writer.add_scalars("DEBUG/diff_nums", \
                          {"0" : np.abs(np.sum(generate_OD_net_numpy == 0) - np.sum(target_graph.OD == 0)),
                           "1" : np.abs(np.sum(generate_OD_net_numpy <= 1) - np.sum(target_graph.OD <= 1)),
                           "5" : np.abs(np.sum(generate_OD_net_numpy <= 5) - np.sum(target_graph.OD <= 5))
                           }, \
                          epoch)
        writer.add_scalar("Test/JSD", JSD, epoch) #  JSD if JSD < 0.31 else 0.31
        writer.add_scalar("Test/RMSE", rmse, epoch) # rmse if rmse < 3000 else 3000
        writer.add_histogram("生成flow_dist", generate_OD_net_numpy[generate_OD_net_numpy <= 30].reshape([-1]), epoch)
        writer.add_histogram("真实flow_dist", target_graph.OD[target_graph.OD <= 30].reshape([-1]), epoch)
        writer.add_image("生成OD", generate_OD_net_numpy.reshape([1, generate_OD_net_numpy.shape[0], generate_OD_net_numpy.shape[1]]), epoch)
        writer.add_image("真实OD", target_graph.OD.reshape([1, target_graph.OD.shape[0], target_graph.OD.shape[1]]), epoch)

        # save some exp info
        logger.log_metrics("RMSE", rmse)
        logger.log_metrics("JSD", JSD)
        if logger.check_training_loss_decent(JSD):
            logger.log_generated_ODNetwork(generate_OD_net_numpy)
            logger.log_best_net(generate_OD_net_numpy)
            logger.log_model_weights(logger.generator_path, generator)
            logger.log_model_weights(logger.discriminator_path, discriminator)
            logger.metrics_to_file()

            writer.add_image("best生成OD", logger.best_jsd_net.reshape([1, generate_OD_net_numpy.shape[0], generate_OD_net_numpy.shape[1]]), epoch)
            writer.add_histogram("best生成flow_dist", generate_OD_net_numpy[generate_OD_net_numpy <= 30].reshape([-1]), epoch)

        if epoch == 10:
            exit(0)







if __name__ == "__main__":

    from utils import MyLogger, gpu_info, narrow_setup, build_graph_from_matrix, skew_sigmoid_numpy, skew_sigmoid_torch, compute_gradient_penalty, one_hot_2_integer, RMSE, prob_vec

    setproctitle.setproctitle("DWGgeneration@rongcan")
    mp.set_start_method('forkserver')

    config = json.load(open("/data/rongcan/code/ODNetGen/config/config_test.json", "r"))
    if config["model"] == "init":
        config["exp"] = config["exp"] + "_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("-", "_")
    elif config["model"] == "load":
        config["exp"] = config["exp"] + "_" + config["IF_load_timestamp"]
    print("\n", "****** configure ******", "\n", config, "\n*************************\n")

    # check GPU available
    if config["check_device"] == 1:
        GPU_no = narrow_setup(interval = 1, mem_need = 10)
        config["device"] = int(GPU_no)
        config["device"] = torch.device("cuda:{}".format(config["device"])) if config["device"] >= 0 else torch.device("cpu")
        print("Using No.", int(GPU_no), "GPU...\n")

    # one random_seed to control the data split
    random.seed(config["random_seed_data"])
    np.random.seed(config["random_seed_data"])
    torch.manual_seed(config["random_seed_data"])
    torch.cuda.manual_seed(config["random_seed_data"])

    logger = MyLogger(config)

    main(config)
