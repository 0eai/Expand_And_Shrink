import copy
import gc
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import *

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict
from sklearn.metrics import accuracy_score

from .models.cnn_mc_mahan import CNN
from .models.cnn_cifar10 import CNN2
from .models.cnn_emnist import CNNEmnist
from .models.two_nn import TwoNN

from .utils import *
from .client import Client

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    def __init__(self, writer, configs):
        self.clients = None
        self._round = 0
        self.writer = writer
        self.configs = configs
        self.model_config = configs.model_config
        self.global_config = configs.global_config
        self.data_config = configs.data_config
        self.init_config = configs.init_config
        self.fed_config = configs.fed_config
        self.client_config = configs.client_config
        self.checkpoint_config = configs.checkpoint_config
        
        # Global Config
        self.seed = self.global_config.seed
        self.mp_flag = self.global_config.is_mp

        # Data Config
        self.data_path = self.data_config.data_path
        self.dataset_name = self.data_config.dataset_name
        # self.num_shards = self.data_config.num_shards
        self.iid = self.data_config.iid

        # Server Config
        self.fraction = self.fed_config.fraction
        self.num_clients = self.fed_config.num_clients
        self.num_rounds = self.fed_config.num_rounds
        self.batch_size = self.fed_config.batch_size

        self.criterion = self.client_config.criterion

        self.ckpt_path = self.checkpoint_config.ckpt_path
        self.ckpt_save_freq = self.checkpoint_config.ckpt_save_freq

        self.device = torch.device(type=self.global_config.device, index=self.global_config.index)

        self.client_participations = np.asarray([0] * self.num_clients)
        self.selected_clients_per_round = list()

    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # split local dataset for each client
        local_datasets, truth_dataset, test_dataset = create_datasets(num_clients=self.num_clients, data_config=self.configs.data_config, writer=self.writer)

        self.model = create_model(self.model_config)

        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.model, self.init_config.gpu_ids)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # assign dataset to each client
        self.clients = self.create_clients(local_datasets, truth_dataset)

        # prepare hold-out dataset for evaluation
        transform = torchvision.transforms.ToTensor()
        if self.dataset_name in ['CIFAR10', 'CIFAR100', 'BelgiumTSC']:
            transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.data = CustomTensorDataset((torch.Tensor(test_dataset.data), torch.Tensor(test_dataset.targets)), transform=transform)
        # self.data = test_dataset
        self.dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())
        
        # configure detailed settings for client upate and 
        self.client_config.dataset_name = self.dataset_name
        
        self.setup_clients(client_config= self.client_config)
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    def create_clients(self, local_datasets, truth_dataset):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, truth_data= truth_dataset, device=self.device, writer=self.writer)
            clients.append(client)
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, client_config):
        """Set up each client."""
        time_diffs = []
        for k, client in tqdm(enumerate(self.clients), leave=False):
            time_diffs.append(client.annotate_data(client_config))
            # client.setup(client_config)
        
        logging.info(f'Average data labeling time: {sum(time_diffs)/len(time_diffs)} seconds')
        self.writer.add_scalar(f'Average data labeling time in seconds', sum(time_diffs)/len(time_diffs), 0)
        
        logging.info(f'Total data labeling time: {sum(time_diffs)} seconds')
        self.writer.add_scalar(f'Total data labeling time in seconds', sum(time_diffs), 0)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())
        self.client_participations[sampled_client_indices] += 1

        self.selected_clients_per_round.append(sampled_client_indices)
        
        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        # message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        # print(message); logging.info(message)
        # del message; gc.collect()

        selected_total_size = 0
        for idx in sampled_client_indices:
            fe_s_t = time.time()
            self.clients[idx].client_update(self._round)
            self.writer.add_scalar(f'Client/{idx}/Training Time Per Round', time.time() - fe_s_t, self._round)
            selected_total_size += len(self.clients[idx])
        # message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        # print(message); logging.info(message)
        # del message; gc.collect()

        return selected_total_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate(self._round)

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def train_federated_model(self):
        fe_s_t = time.time()
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_selected_clients(sampled_client_indices)
        self.writer.add_scalar('Server/Training Time Per Round', time.time() - fe_s_t, self._round)
        
        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)

        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""

        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                # print('data:', data.size(), labels.size())
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                
                # if self.device.type == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        return test_loss, test_accuracy

    def fit(self):
        ckpt_path = join(self.ckpt_path, self.model_config.name, self.data_config.dataset_name)
        
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}
        #log_histogram(self.writer, self.model , 0)
        for r in range(self.num_rounds):
            fe_s_t = time.time()
            self._round = r + 1
            
            self.train_federated_model()
            test_loss, test_accuracy = self.evaluate_global_model()
            # log_histogram(self.writer, self.model , r + 1)
            
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)
            
            self.writer.add_scalar('Loss', test_loss, self._round)
            self.writer.add_scalar('Accuracy', test_accuracy, self._round)

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()
            self.writer.add_scalar('Server/Total Time Per Round', time.time() - fe_s_t, self._round)

        losses = self.results['loss']
        accuracies = self.results['accuracy']
        self.writer.add_scalar('Low/Loss', losses[np.argmin(losses)], np.argmin(losses) + 1)
        self.writer.add_scalar('Max/Accuracy', accuracies[np.argmax(accuracies)], np.argmax(accuracies) + 1)
        
        for i, ctr in enumerate(self.client_participations):
            self.writer.add_scalar('Client/Each Client Participations in Training', ctr, i + 1)
        self.transmit_model()
        
