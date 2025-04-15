import os
from os.path import join
import argparse
import logging
import omegaconf
from torchvision.transforms import *
from torch.utils.tensorboard import SummaryWriter

from src.server import Server

parser = argparse.ArgumentParser()
parser.add_argument("-ds", "--dataset", type=str, help = "dataset name")
parser.add_argument("-i", "--iid", type=bool, help = "data iid / non-iid flag")
parser.add_argument("-lf", "--lbl_fraction", type=float, help = "truth data fraction")
parser.add_argument("-k", "--n_client", type=int, help = "number of clients")
parser.add_argument("-c", "--client_fraction", type=float, help = "client fraction")
parser.add_argument("-cl", "--n_cluster", type=int, help = "number of clusters")
parser.add_argument("-dt", "--device_type", type=str, help = "device type")
parser.add_argument("-di", "--device_index", type=int, help = "device index")


def print_in_format(cfg, writer):
    for key, val in cfg.items():
        if type(val) == type(cfg):
            print(key, ':') 
            for sub_key, sub_val in val.items():
                writer.add_text(key, str(sub_key) + ' : ' + str(sub_val)); print('    ', sub_key, ':', sub_val)
        else:
            writer.add_text(key, ' : ' + str(val)); print(key, ':', val)


def run_experiment(configs):
    log_config = configs.log_config

    dataset_name = configs.data_config.dataset_name
    model_name = configs.model_config.name
    iid = configs.data_config.iid
    lbl_fraction = configs.data_config.lbl_fraction
    clusters = configs.client_config.num_clusters
    num_clients = configs.fed_config.num_clients
    fraction = configs.fed_config.fraction

    # modify log_path to contain current time

    log_config.log_path = join(log_config.log_path, f"{model_name}_{dataset_name}_{iid}_{lbl_fraction} c_{clusters}, k_{num_clients} c_{fraction}")

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_config.log_path)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config.log_path, log_config.log_name),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")
    
    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)

    print_in_format(configs, writer=writer)

    # initialize federated learning 
    central_server = Server(writer, configs)
    central_server.setup()
    
    # do federated learning
    # central_server.fit()

    '''
    writer.add_hparams(
            {"iid": self.data_config.iid, "lbl_fraction": self.data_config.lbl_fraction, "num_clients": self.fed_config.num_clients, "fraction": self.fed_config.fraction, "num_clusters": self.client_config.num_clusters},
            {"accuracy": test_accuracy, "loss": test_loss,},)'''

    # save resulting losses and metrics
    #with open(os.path.join(log_config.log_path, "result.pkl"), "wb") as f:
    #    pickle.dump(central_server.results, f)
    
    # bye!
    message = "...done all learning process!\n"
    print(message); logging.info(message)
    return 0


if __name__ == "__main__":
    # read configuration file
    cfg_path = './configs/expand_and_shrink.yaml'
    configs = omegaconf.OmegaConf.load(cfg_path)
    log_path = configs.log_config.log_path

    args = parser.parse_args()
    
    dataset = args.dataset

    configs.data_config.dataset_name = dataset
    configs.data_config.dataset_name = configs.datasets.get(dataset).name
    configs.data_config.params = configs.datasets.get(dataset).params

    if dataset in ['MNIST', 'FashionMNIST']:
        configs.model_config = configs.models.TwoNN
    elif dataset == 'EMNIST':
        configs.model_config = configs.models.CNN 
        configs.model_config.num_classes = 47
    elif dataset == 'CIFAR10':
        configs.model_config = configs.models.ResNet9
    elif dataset == 'CIFAR100':
        configs.model_config = configs.models.CNN2
        configs.model_config.num_classes = 100
    elif dataset == 'BelgiumTSC':
        configs.model_config = configs.models.CNN2
        configs.model_config.num_classes = 62

    configs.data_config.iid = args.iid
    configs.data_config.lbl_fraction = args.lbl_fraction
    configs.fed_config.num_clients = args.n_client
    configs.fed_config.fraction = args.client_fraction
    configs.client_config.num_clusters = args.n_cluster
    configs.log_config.log_path = log_path
    configs.global_config.device = args.device_type
    configs.global_config.index = args.device_index
    print('Starting experiment on device: ', args.device_type, args.device_index)
    run_experiment(configs)
