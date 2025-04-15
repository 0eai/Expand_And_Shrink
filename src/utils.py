import os
from os import listdir
from os.path import isfile, join
import logging
import omegaconf
import numpy as np
import pandas as pd
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import shutil
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, KMeans

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets import *
import torchvision
from torchvision.datasets import ImageFolder
from transformers import BeitForImageClassification, AutoFeatureExtractor

from .models.cnn_mc_mahan import CNN
from .models.cnn_cifar10 import CNN2
from .models.cnn_emnist import CNNEmnist
from .models.two_nn import TwoNN
from .models.resnet9 import ResNet9
logger = logging.getLogger(__name__)

torch.manual_seed(42)

#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

#########################
#     Create Model      #
#########################

def create_model(model_config, class_to_idx = None):
    print(f"Creating model: {model_config.name}" )
    return eval(model_config.name)(**model_config)

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    return model

def save_checkpoint(model, path):
    """
    Save a checkpoint specific to Data2Vec
    Args:
        model: a nn.Module instance
        optimizer
        path: path to save checkpoint to
        epoch_num: current epoch number
        save_freq: save frequency based on epoch number

    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f'best_model.pt')
    
    checkpoint = {'TwoNN': model.state_dict()}
    torch.save(checkpoint, path)
    print(f'Saved checkpoint to `{path}`')

def save_finetune_checkpoint(model, path):
    """
    Save a checkpoint specific to Data2Vec
    Args:
        model: a nn.Module instance
        optimizer
        path: path to save checkpoint to
        epoch_num: current epoch number
        save_freq: save frequency based on epoch number

    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f'best_model.pt')
    
    checkpoint = {'beit': model.state_dict()}
    torch.save(checkpoint, path)
    print(f'Saved checkpoint to `{path}`')

#################
# Dataset split #
#################
class CustomDataset():
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        #print(y)
        return x.float(), y.long()

    def __len__(self):
        return self.tensors[0].size(0)

def create_centerlized_tsc_dataset(data_config):
    dataset_name = data_config.dataset_name
    params = data_config.params
    lbl_fraction = data_config.lbl_fraction

    root_dir = 'data/BelgiumTSC'

    train_sub_directory = 'Training'
    test_sub_directory = 'Testing'
    train_csv_file_name = 'train_data.csv'
    test_csv_file_name = 'test_data.csv'

    train_csv_file_path = os.path.join(root_dir, train_sub_directory, train_csv_file_name)
    test_csv_file_path = os.path.join(root_dir, test_sub_directory, test_csv_file_name)

    train_csv_data = pd.read_csv(train_csv_file_path)
    test_csv_data = pd.read_csv(test_csv_file_path)

    train_data = []
    train_targets = []
    test_data = []
    test_targets = []
    for idx in range(len(train_csv_data)):
        img_path = os.path.join(root_dir, train_sub_directory, train_csv_data.iloc[idx, 0])
        img = np.array(Image.open(img_path).resize((32, 32)))
        classId = train_csv_data.iloc[idx, 1]
        train_data.append(img)
        train_targets.append(classId)

    for idx in range(len(test_csv_data)):
        img_path = os.path.join(root_dir, test_sub_directory, test_csv_data.iloc[idx, 0])
        img = np.array(Image.open(img_path).resize((32, 32)))
        classId = test_csv_data.iloc[idx, 1]
        test_data.append(img)
        test_targets.append(classId)

    train_data, train_targets = np.asarray(train_data), np.asarray(train_targets)
    test_data, test_targets = np.asarray(test_data), np.asarray(test_targets)
    print(train_data.shape, train_targets.shape, test_data.shape, test_targets.shape)

    shape = [0]
    shape.extend(list(train_data.shape[1:]))
    truth_data = np.empty(shape=shape)
    truth_targets = []

    targets, counts = np.unique(train_targets, return_counts=True)
    for target, count in zip(targets, counts):
        data_idx = np.where(train_targets == target)
        data_idx = data_idx[0]        
            
        num_lbl = int(math.ceil(count * lbl_fraction))
        lbl_idx = data_idx[:num_lbl]
        truth_data = np.append(truth_data, np.take(train_data, lbl_idx, axis=0), axis=0)
        truth_targets = np.append(truth_targets, np.take(train_targets, lbl_idx))
            
        train_data = np.delete(train_data, lbl_idx, axis=0)
        train_targets = np.delete(train_targets, lbl_idx)

    train_targets = train_targets.astype('int64')
    truth_targets = truth_targets.astype('int64')
    return (train_data, train_targets), (truth_data, truth_targets), CustomDataset(test_data, test_targets)


def create_centerlized_datasets(data_config):
    if data_config.dataset_name == 'BelgiumTSC':
        return create_centerlized_tsc_dataset(data_config)

    # Split the whole dataset in IID or non-IID manner for distributing to clients.
    dataset_name = data_config.dataset_name
    params = data_config.params
    lbl_fraction = data_config.lbl_fraction

    # get dataset from torchvision.datasets if exists
    if hasattr(torchvision.datasets, dataset_name):
        # prepare raw training & test datasets
        training_dataset = eval(dataset_name)(**params)
        params.train = False 
        test_dataset = eval(dataset_name)(**params)
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets
    if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)

    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "ndarray" not in str(type(training_dataset.targets)):
        training_dataset.targets = np.asarray(training_dataset.targets)

    test_dataset.targets = np.array(test_dataset.targets)

    shuffled_indices = torch.randperm(len(training_dataset))
    training_dataset.data = training_dataset.data[shuffled_indices]
    training_dataset.targets = training_dataset.targets[shuffled_indices]    
    
    shape = [0]
    shape.extend(list(training_dataset.data.shape[1:]))
    truth_data = np.empty(shape=shape)
    truth_data_lbl = []

    targets, counts = np.unique(training_dataset.targets, return_counts=True)
    for target, count in zip(targets, counts):
        data_idx = np.where(training_dataset.targets == target)
        data_idx = data_idx[0]        
        
        num_lbl = int(count * lbl_fraction)
        lbl_idx = data_idx[:num_lbl]
        truth_data = np.append(truth_data, np.take(training_dataset.data, lbl_idx, axis=0), axis=0)
        truth_data_lbl = np.append(truth_data_lbl, np.take(training_dataset.targets, lbl_idx))
        
        training_dataset.data = np.delete(training_dataset.data, lbl_idx, axis=0)
        training_dataset.targets = np.delete(training_dataset.targets, lbl_idx)

    training_dataset.targets = training_dataset.targets.astype('int64')
    truth_data_lbl = truth_data_lbl.astype('int64')

    return (training_dataset.data, training_dataset.targets), (truth_data, truth_data_lbl), test_dataset

def create_fed_tsc_datasets(num_clients, data_config, writer=None):
    dataset_name = data_config.dataset_name
    params = data_config.params
    lbl_fraction = data_config.lbl_fraction
    iid = data_config.iid
    num_shards = data_config.num_shards

    root_dir = 'data/BelgiumTSC'

    train_sub_directory = 'Training'
    test_sub_directory = 'Testing'
    train_csv_file_name = 'train_data.csv'
    test_csv_file_name = 'test_data.csv'

    train_csv_file_path = os.path.join(root_dir, train_sub_directory, train_csv_file_name)
    test_csv_file_path = os.path.join(root_dir, test_sub_directory, test_csv_file_name)

    train_csv_data = pd.read_csv(train_csv_file_path)
    test_csv_data = pd.read_csv(test_csv_file_path)

    train_data = []
    train_targets = []
    test_data = []
    test_targets = []
    for idx in range(len(train_csv_data)):
        img_path = os.path.join(root_dir, train_sub_directory, train_csv_data.iloc[idx, 0])
        img = np.array(Image.open(img_path).resize((32, 32)))
        classId = train_csv_data.iloc[idx, 1]
        train_data.append(img)
        train_targets.append(classId)

    for idx in range(len(test_csv_data)):
        img_path = os.path.join(root_dir, test_sub_directory, test_csv_data.iloc[idx, 0])
        img = np.array(Image.open(img_path).resize((32, 32)))
        classId = test_csv_data.iloc[idx, 1]
        test_data.append(img)
        test_targets.append(classId)

    train_data, train_targets = np.asarray(train_data), np.asarray(train_targets)
    test_data, test_targets = np.asarray(test_data), np.asarray(test_targets)
    log_data_distribution(train_targets, writer=writer)
    shape = [0]
    shape.extend(list(train_data.shape[1:]))
    truth_data = np.empty(shape=shape)
    truth_targets = []

    targets, counts = np.unique(train_targets, return_counts=True)
    for target, count in zip(targets, counts):
        data_idx = np.where(train_targets == target)
        data_idx = data_idx[0]        
            
        num_lbl = int(math.ceil(count * lbl_fraction))
        lbl_idx = data_idx[:num_lbl]
        truth_data = np.append(truth_data, np.take(train_data, lbl_idx, axis=0), axis=0)
        truth_targets = np.append(truth_targets, np.take(train_targets, lbl_idx))
            
        train_data = np.delete(train_data, lbl_idx, axis=0)
        train_targets = np.delete(train_targets, lbl_idx)

    log_training_data_distribution(train_targets, writer=writer)
    log_truth_data_distribution(truth_targets, writer=writer)
    train_targets = train_targets.astype('int64')
    truth_targets = truth_targets.astype('int64')
    
    if iid:
        split_datasets = list(
            zip(
                np.array_split(train_data, num_clients),
                np.array_split(train_targets, num_clients)
            )
        )
    
    else:
        sorted_indices = np.argsort(train_targets)
        train_data = train_data[sorted_indices]
        train_targets = train_targets[sorted_indices]
        
        shard_size = len(train_data) // num_shards #300

        shard_inputs = np.asarray(np.array_split(train_data, shard_size), dtype=object)
        shard_labels = np.asarray(np.array_split(train_targets, shard_size), dtype=object)

        shuffled_indices = torch.randperm(len(shard_inputs))
        shard_inputs = shard_inputs[shuffled_indices]
        shard_labels = shard_labels[shuffled_indices]

        training_inputs = np.array(np.array_split(shard_inputs, num_clients), dtype=object)
        training_labels = np.array(np.array_split(shard_labels, num_clients), dtype=object)

        split_datasets = []
        for i, (inputs, lables) in enumerate(zip(training_inputs, training_labels)):
            client_data = np.empty(shape=shape)
            client_lbl = []
            for j, (training_input, training_label) in enumerate(zip(inputs, lables)):
                client_data = np.append(client_data, training_input, axis=0)
                client_lbl = np.append(client_lbl, training_label)
            split_datasets.append((client_data, client_lbl))

    log_clients_data_distribution(split_datasets, writer=writer)
    return split_datasets, (truth_data, truth_targets), CustomDataset(test_data, test_targets)


def create_datasets(num_clients, data_config, writer=None):
    if data_config.dataset_name == 'BelgiumTSC':
        return create_fed_tsc_datasets(num_clients, data_config, writer)
    
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    dataset_name = data_config.dataset_name
    params = data_config.params
    lbl_fraction = data_config.lbl_fraction
    iid = data_config.iid
    num_shards = data_config.num_shards
    # get dataset from torchvision.datasets if exists
    if hasattr(torchvision.datasets, dataset_name):
        # prepare raw training & test datasets
        training_dataset = eval(dataset_name)(**params)
        params.train = False 
        test_dataset = eval(dataset_name)(**params)

    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets
    if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)
    num_categories = np.unique(training_dataset.targets).shape[0]

    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "ndarray" not in str(type(training_dataset.targets)):
        training_dataset.targets = np.asarray(training_dataset.targets)

    test_dataset.targets = np.array(test_dataset.targets)

    shuffled_indices = torch.randperm(len(training_dataset))
    training_dataset.data = training_dataset.data[shuffled_indices]
    training_dataset.targets = training_dataset.targets[shuffled_indices]
    log_data_distribution(training_dataset.targets, writer=writer)
    
    
    shape = [0]
    shape.extend(list(training_dataset.data.shape[1:]))
    truth_data = np.empty(shape=shape)
    truth_data_lbl = []

    targets, counts = np.unique(training_dataset.targets, return_counts=True)
    for target, count in zip(targets, counts):
        data_idx = np.where(training_dataset.targets == target)
        data_idx = data_idx[0]        
        
        num_lbl = int(count * lbl_fraction)
        lbl_idx = data_idx[:num_lbl]
        truth_data = np.append(truth_data, np.take(training_dataset.data, lbl_idx, axis=0), axis=0)
        truth_data_lbl = np.append(truth_data_lbl, np.take(training_dataset.targets, lbl_idx))
        
        training_dataset.data = np.delete(training_dataset.data, lbl_idx, axis=0)
        training_dataset.targets = np.delete(training_dataset.targets, lbl_idx)
    log_training_data_distribution(training_dataset.targets, writer=writer)
    log_truth_data_distribution(truth_data_lbl, writer=writer)

    training_dataset.targets = training_dataset.targets.astype('int64')
    truth_data_lbl = truth_data_lbl.astype('int64')

    if iid:
        split_datasets = list(
            zip(
                np.array_split(training_dataset.data, num_clients),
                np.array_split(training_dataset.targets, num_clients)
            )
        )
    
    else:
        sorted_indices = np.argsort(training_dataset.targets)
        training_dataset.data = training_dataset.data[sorted_indices]
        training_dataset.targets = training_dataset.targets[sorted_indices]
        
        shard_size = len(training_dataset) // num_shards #300

        shard_inputs = np.asarray(np.array_split(training_dataset.data, shard_size), dtype=object)
        shard_labels = np.asarray(np.array_split(training_dataset.targets, shard_size), dtype=object)

        shuffled_indices = torch.randperm(len(shard_inputs))
        shard_inputs = shard_inputs[shuffled_indices]
        shard_labels = shard_labels[shuffled_indices]

        training_inputs = np.array(np.array_split(shard_inputs, num_clients), dtype=object)
        training_labels = np.array(np.array_split(shard_labels, num_clients), dtype=object)

        split_datasets = []
        for i, (inputs, lables) in enumerate(zip(training_inputs, training_labels)):
            client_data = np.empty(shape=shape)
            client_lbl = []
            for j, (training_input, training_label) in enumerate(zip(inputs, lables)):
                client_data = np.append(client_data, training_input, axis=0)
                client_lbl = np.append(client_lbl, training_label)
            split_datasets.append((client_data, client_lbl))

    log_clients_data_distribution(split_datasets, writer=writer)
    
    test_dataset.data = torch.Tensor(test_dataset.data).to(torch.float)
    return split_datasets, (truth_data, truth_data_lbl), test_dataset

def log_data_distribution(targets, writer):
    for target in np.unique(targets):
        data_idx = np.where(targets == target)
        writer.add_scalar('Data/Total Samples Per Class', len(data_idx[0]), target)
        # print(f"\t target {target}: {len(data_idx[0])}")

def log_training_data_distribution(targets, writer):
    for target in np.unique(targets):
        data_idx = np.where(targets == target)
        writer.add_scalar('Data/Training Samples Per Class', len(data_idx[0]), target)
        # print(f"\t target {target}: {len(data_idx[0])}")

def log_truth_data_distribution(targets, writer):
    for target in np.unique(targets):
        data_idx = np.where(targets == target)
        writer.add_scalar('Data/Truth Samples Per Class', len(data_idx[0]), target)
        # print(f"\t target {target}: {len(data_idx[0])}")

def log_clients_data_distribution(split_datasets, writer):
    total = 0
    for i, (data, targets) in enumerate(split_datasets):
        count = 0
        for target in np.unique(targets):
            data_idx = np.where(targets == target)
            writer.add_scalar(f'Client/{i}/Training Samples Per Class', len(data_idx[0]), target)
            # print(f"\t\t target {target}: {len(data_idx[0])}")
        total = total + len(targets)
        writer.add_scalar(f'Data/Total Training Samples Per Client', len(targets), i)
    # print(f'total: {total}')


def log_histogram(writer, model, round, id=None):
    for name, weight in model.named_parameters():
        if id is not None:
            writer.add_histogram(f'Client {id}: {name}',weight, round)
            writer.add_histogram(f'Client {id}: {name}.grad',weight.grad, round)
        else:
            writer.add_histogram(f'Global : {name}',weight, round)
            # writer.add_histogram(f'Global : {name}.grad',weight.grad, round)

def infer_cluster_labels(kmeans, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        # index = np.where(kmeans.labels_ == i)
        index = np.where(kmeans.labels_[:len(actual_labels)] == i)

        if len(index[0]) == 0:
            continue

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])
        
        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]
        
    return inferred_labels

def annotate_data(data, truth_data, num_clusters, dataset_name):
    shape = list(data[0].shape[1:])
        
    x_train = data[0].reshape(len(data[0]), -1).astype('uint8')
    y_train = data[1].astype('int64')
        
    x_truth = truth_data[0].reshape(len(truth_data[0]), -1)
    y_truth = truth_data[1]

    X = np.concatenate((x_truth, x_train), axis=0)
    Y = np.concatenate((y_truth, y_train), axis=0)

    kmeans_on_train_data = MiniBatchKMeans(n_clusters = num_clusters)
    # kmeans_on_train_data = MiniBatchKMeans(n_clusters = len(x_truth))
    # kmeans_on_train_data = KMeans(init='k-means++', n_clusters=len(x_truth), n_init=10)
    print('X: ', max(X[0]))
    print('X: ', X.shape)
    if dataset_name in ['CIFAR10', 'CIFAR100', 'BelgiumTSC']:
        X_ = X.reshape(X.shape[0], 32, 32, 3).mean(3).reshape(X.shape[0], -1)
        X_ = X_.astype('uint8')
        print('X_: ', X_.shape)
        print('X_: ', max(X_[0]))
        # Fitting the model to training set
        kmeans_on_train_data.fit(X_)
    else:
        kmeans_on_train_data.fit(X)
        
    reference_labels = infer_cluster_labels(kmeans_on_train_data, y_truth)
    # print('reference_labels: ', reference_labels)
    centroids = kmeans_on_train_data.cluster_centers_
        
    y_pred = np.empty(len(Y))
    for i, centriod in enumerate(centroids):
        reference_label = None
        for key, value in reference_labels.items():
            if i in value:
                reference_label = key
        indexes = np.where(kmeans_on_train_data.labels_ == i)
        # print(f'cluster {i} y_pred len {len(y_pred)} reference_label {reference_label}', len(indexes[0]))
        y_pred[indexes[0]] = reference_label
    y_pred = y_pred.astype('int64')
        
    # print('\n\nCounter self.data[1]: ', Counter(Y))
    # print('Counter y_pred: ', Counter(y_pred))
    
    accuracy = accuracy_score(Y, y_pred)
    homogeneity_score = metrics.homogeneity_score(Y, y_pred)
    print('_'*50)

    shape = list(data[0].shape)
    shape[0] = len(X)
    X = X.reshape(shape)
    return (torch.Tensor(X), torch.Tensor(y_pred)), accuracy, homogeneity_score

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    cfg_path = '../config.yaml'
    configs = omegaconf.OmegaConf.load(cfg_path)
    local_datasets, test_dataset = create_datasets(configs.fed_config.num_clients, configs.data_config)
    
    print(len(local_datasets))