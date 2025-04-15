import time
from itertools import product
import numpy
from experiment import Experiment
import numpy as np
import torch
import subprocess


seed = 42
np.random.seed(seed)
torch.manual_seed(42)

if __name__ == '__main__':

    device_type = 'cuda'
    parameters = dict(
        client_fractions = [0.5], 
        lbl_fractions = [0.01, 0.03, 0.05, 0.07, 0.09], # 0.01, 0.03, 0.05, 0.07, 0.09
        dataset_names = ['MNIST', 'FashionMNIST', 'CIFAR10'], # 'MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'BelgiumTSC'
        iid_flags = [True],
        clients = [100],
        clusters = [(10, 47, 100, 62), (20, 94, 200, 124), (40, 188, 400, 248), (80, 376, 800, 496), (160, 752, 1600, 992), (0, 0, 0, 0)] # 
    )

    param_values = [v for v in parameters.values()]
    for run_id, (client_fraction, lbl_fraction, dataset, iid_flag, num_clients, num_clusters) in enumerate(product(*param_values)):
        clusters = num_clusters[0]
        if dataset == 'EMNIST':
            clusters = num_clusters[1]
        elif dataset == 'CIFAR100':
            clusters = num_clusters[2]
        elif dataset == 'BelgiumTSC':
            clusters = num_clusters[3]
        
        cmd = f'python main.py -ds {dataset} -i {iid_flag} -lf {lbl_fraction} -k {num_clients} -c {client_fraction} -cl {clusters} -dt {device_type} -di 0'
        
        p = subprocess.Popen(cmd, shell=True)
        print(f'Experiment added: id {run_id}, {dataset}, {iid_flag}, {lbl_fraction}, {num_clients}, {client_fraction}, {clusters}')
        while True:
            time.sleep(30)
            if p is not None and p.poll() is not None:
                break
            