import gc
import pickle
import logging
import time
from torchvision.transforms import *

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn import metrics
from collections import Counter
from multiprocessing import cpu_count
from sklearn.cluster import MiniBatchKMeans

from .utils import *
from torch.utils.data import DataLoader
import time

logger = logging.getLogger(__name__)

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

def get_H(RH):
    H = []
    for i in range(256):
        if i in RH.keys():
            H.append(RH[i])
        else:
            H.append(0)
    return H

def L2Norm(H1,H2):
    distance =0
    for i in range(len(H1)):
        distance += np.square(H1[i]-H2[i])
    return np.sqrt(distance)

class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, truth_data, device, writer):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.truth_data = truth_data
        self.device = device
        self.writer = writer
        self.__model = None

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, client_config):
        """Set up common configuration of each client; called by center server."""
        self.client_config = client_config
        self.num_clusters = client_config.num_clusters
        # self.annotate_data(client_config)
        
        transform = torchvision.transforms.ToTensor()
        if client_config.dataset_name in ['CIFAR10', 'CIFAR100', 'BelgiumTSC']:
            transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.data = CustomTensorDataset(self.data, transform=transform)
        self.dataloader = DataLoader(self.data, batch_size=client_config.batch_size, shuffle=True, num_workers=cpu_count())
        
        self.local_epochs = client_config.local_epochs
        self.criterion = client_config.criterion
        self.optimizer = client_config.optimizer
        self.optim_config = client_config.optim_config
        self.criterion_config = client_config.criterion_config

        # self.log_dataset()
        
    def log_dataset(self):
        images, labels = next(iter(self.dataloader))
        grid = torchvision.utils.make_grid(images, nrow=4)
        for i, (data, labels) in enumerate(self.dataloader):
            self.writer.add_image(f"Client {self.id} train images", grid, i)
        # self.writer.add_graph(self.model, images)

    def annotate_data(self, client_config):
        self.client_config = client_config
        self.num_clusters = client_config.num_clusters
        
        shape = list(self.data[0].shape[1:])
        #shape.extend())
        #truth_data = np.empty(shape=shape)
        
        x_train = self.data[0].reshape(len(self.data[0]), -1).astype('uint8')
        y_train = self.data[1].astype('int64')
        
        x_truth = self.truth_data[0].reshape(len(self.truth_data[0]), -1)
        y_truth = self.truth_data[1]

        X = np.concatenate((x_truth, x_train), axis=0)
        Y = np.concatenate((y_truth, y_train), axis=0)
        start_time = time.time()
        if self.num_clusters == 0:
            kmeans_on_train_data = MiniBatchKMeans(n_clusters = len(x_truth))
        else:
            kmeans_on_train_data = MiniBatchKMeans(n_clusters = self.num_clusters)

        # kmeans_on_train_data = KMeans(init='k-means++', n_clusters=len(x_truth), n_init=10, n_jobs=cpu_count())
        # Fitting the model to training set
        # print('X: ', max(X[0]))
        # print('X: ', X.shape)
        if self.client_config.dataset_name in ['CIFAR10', 'CIFAR100', 'BelgiumTSC']:
            X_ = X.reshape(X.shape[0], 32, 32, 3).mean(3).reshape(X.shape[0], -1)
            X_ = X_.astype('uint8')
            # print('X_: ', X_.shape)
            # print('X_: ', max(X_[0]))
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
        time_diff = time.time() - start_time
        
        logging.info(f'Client {self.id} data labeling time: {time_diff} seconds')
        self.writer.add_scalar(f'Data/Labeling time in seconds', time_diff, self.id)
        
        # print('\n\nCounter self.data[1]: ', Counter(Y))
        # print('Counter y_pred: ', Counter(y_pred))

        self.writer.add_scalar(f'Data/Homogeneity Score(kmeans)', metrics.homogeneity_score(Y, kmeans_on_train_data.labels_), self.id)
        self.writer.add_scalar(f'Data/Label Accuracy(kmeans)', accuracy_score(Y, kmeans_on_train_data.labels_), self.id)
        self.writer.add_scalar(f'Data/Homogeneity Score', metrics.homogeneity_score(Y, y_pred), self.id)
        self.writer.add_scalar(f'Data/Label Accuracy', accuracy_score(Y, y_pred), self.id)

        shape = list(self.data[0].shape)
        shape[0] = len(X)
        X = X.reshape(shape)
        self.data = (torch.Tensor(X), torch.Tensor(y_pred))
        return time_diff

    def client_update(self, round):
        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)

        for e in range(self.local_epochs):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step() 

                # if self.device == "cuda": torch.cuda.empty_cache()               
        self.model.to("cpu")
        # log_histogram(self.writer, self.model , round, id=self.id)

    def client_evaluate(self,round):
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                # if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation for round {str(round).zfill(4)}!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy
