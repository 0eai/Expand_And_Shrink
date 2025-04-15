import os
import time
import logging
import omegaconf
from torchvision.transforms import *
from itertools import product
from pathlib import Path


from torch.utils.tensorboard import SummaryWriter

from src.centerlized import Centerlized

def print_in_format(cfg, writer):
    for key, val in cfg.items():
        if type(val) == type(cfg):
            print(key, ':') 
            for sub_key, sub_val in val.items():
                writer.add_text(key, str(sub_key) + ' : ' + str(sub_val)); print('    ', sub_key, ':', sub_val)
        else:
            writer.add_text(key, ' : ' + str(val)); print(key, ':', val)

def run_centerlized_experiment(configs):
    log_config = configs.log_config

    # modify log_path to contain current time
    log_config.log_path = f'{log_config.log_path}/tb/cent/{configs.data_config.dataset_name}/{configs.data_config.iid}/{configs.data_config.lbl_fraction}/{configs.client_config.num_clusters}'
    Path(log_config.log_path).mkdir(parents=True, exist_ok=True)

    # initiate TensorBaord for tracking losses and metrics
    writer = None # SummaryWriter(log_dir=log_config.log_path, filename_suffix="FL")
    
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

    # print_in_format(configs, writer=writer)

    # initialize centerlized learning 
    central = Centerlized(writer, configs)
    central.setup()
    
    # do centerlized learning
    central.fit()

    # bye!
    message = "...done all learning process!\n"
    print(message); logging.info(message)

if __name__ == "__main__":
    # read configuration file
    cfg_path = './configs/expand_and_shrink.yaml'
    configs = omegaconf.OmegaConf.load(cfg_path)
    log_path = configs.log_config.log_path

    parameters = dict(
        lbl_fractions = [0.01, 0.03, 0.05, 0.07, 0.09], # [0.01, 0.03, 0.05, 0.07, 0.09],
        dataset_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'BelgiumTSC'], # 'CIFAR10', 'CIFAR100'
        iid_flags = [True],
        clusters = [(10, 47, 100, 62), (20, 94, 200, 124), (40, 188, 400, 248), (80, 376, 800, 496), (160, 752, 1600, 992)] 
    )
    param_values = [v for v in parameters.values()]
    
    for lbl_fraction, dataset, iid_flag, num_clusters in product(*param_values):
        print(dataset, iid_flag, lbl_fraction, num_clusters)

        clusters = num_clusters[0]
        configs.data_config.dataset_name = dataset
        configs.data_config.dataset_name = configs.datasets.get(dataset).name
        configs.data_config.params = configs.datasets.get(dataset).params

        if dataset in ['MNIST', 'FashionMNIST']:
            configs.model_config = configs.models.TwoNN
        elif dataset == 'EMNIST':
            configs.model_config = configs.models.TwoNN # CNN
            configs.model_config.num_classes = 47
            clusters = num_clusters[1]
        elif dataset == 'CIFAR10':
            configs.model_config = configs.models.CNN2
        elif dataset == 'CIFAR100':
            configs.model_config = configs.models.CNN2
            configs.model_config.num_classes = 100
            clusters = num_clusters[2]
        elif dataset == 'BelgiumTSC':
            configs.model_config = configs.models.CNN2
            configs.model_config.num_classes = 62
            clusters = num_clusters[3]

        configs.data_config.iid = iid_flag
        configs.data_config.lbl_fraction = lbl_fraction
        configs.client_config.num_clusters = clusters
        configs.cent_config.num_clusters = clusters
        configs.log_config.log_path = log_path

        run_centerlized_experiment(configs)

    # bye!
    message = "...done all experiments!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); exit()

