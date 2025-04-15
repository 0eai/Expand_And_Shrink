import time
from itertools import product
import numpy
from experiment import Experiment
import numpy as np
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(42)

if __name__ == '__main__':

    device_type = 'cuda'
    parameters = dict(
        client_fractions = [0.5], 
        lbl_fractions = [0.05], # 0.01, 0.03, 0.05, 0.07, 0.09
        dataset_names = ['CIFAR10'], # 'MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'BelgiumTSC'
        iid_flags = [True],
        clients = [100],
        clusters = [(160, 752, 1600, 992), (0, 0, 0, 0)] # 
    )

    executer = Experiment(exps_per_dev=1)
    param_values = [v for v in parameters.values()]
    for run_id, (client_fraction, lbl_fraction, dataset, iid_flag, num_clients, num_clusters) in enumerate(product(*param_values)):
        clusters = num_clusters[0]
        if dataset == 'EMNIST':
            clusters = num_clusters[1]
        elif dataset == 'CIFAR100':
            clusters = num_clusters[2]
        elif dataset == 'BelgiumTSC':
            clusters = num_clusters[3]
        
        cmd = f'python main.py -ds {dataset} -i {iid_flag} -lf {lbl_fraction} -k {num_clients} -c {client_fraction} -cl {clusters} -dt {device_type} -di '
        
        executer.add_experiment(exp=cmd)
        print(f'Experiment added: id {run_id}, {dataset}, {iid_flag}, {lbl_fraction}, {num_clients}, {client_fraction}, {clusters}')
    executer.run()

'''
    parameters = dict(
        dataset_names = ['MNIST'], # 'MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'BelgiumTSC'
        iid_flags = [True],
        lbl_fractions = [0.05, 0.07, 0.09], # 0.01, 0.03, 0.05, 0.07, 0.09
        clients = [100],
        client_fractions = [0.5], # 
        clusters = [(10, 47, 100, 62), (20, 94, 200, 124), (40, 188, 400, 248), (80, 376, 800, 496), (160, 752, 1600, 992)]
    )

    parameters = dict(
        dataset_names = ['MNIST'], # 'MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'BelgiumTSC'
        iid_flags = [True],
        lbl_fractions = [0.07, 0.09], # 0.01, 0.03, 0.05, 0.07, 0.09
        clients = [100],
        client_fractions = [0.1, 0.2, 0.3, 0.4], # 
        clusters = [(10, 47, 100, 62), (20, 94, 200, 124), (40, 188, 400, 248), (80, 376, 800, 496), (160, 752, 1600, 992)]
    )
'''