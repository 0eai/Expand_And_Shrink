import gc
import time
import numpy as np
import torchvision
import torch
import omegaconf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import *

from torchvision.datasets import *
from torchvision.transforms import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from .utils import *
from .datamodule import DataModule
from .litmodel import LitModel


logger = logging.getLogger(__name__)

def print_data_distribution(targets):
    print("*"*50)
    for target in np.unique(targets):
        data_idx = np.where(targets == target)
        print(f"\t target {target}: {len(data_idx[0])}")
    print("*"*50)

class MetricTracker(Callback):

  def __init__(self):
    self.epochs = []
    self.train_losses = []
    self.val_losses = []
    self.val_accs = []
    self.metrics = []


  def on_validation_epoch_end(self, trainer, module):
    elogs = trainer.logged_metrics # access it here
    # self.train_losses = elogs['train_loss']
    self.val_losses.append(elogs['val_loss'])
    self.val_accs.append(elogs['val_acc'])
    self.metrics.append(elogs)
    # do whatever is needed

class Centerlized(object):
    def __init__(self, writer, configs):
        self.writer = writer
        self.configs = configs
        self.model_config = configs.model_config
        self.global_config = configs.global_config
        self.data_config = configs.data_config
        self.init_config = configs.init_config
        self.cent_config = configs.cent_config
        self.client_config = configs.client_config
        self.checkpoint_config = configs.checkpoint_config
        
        # Global Config
        self.seed = self.global_config.seed
        self.device = self.global_config.device
        self.mp_flag = self.global_config.is_mp

        # Data Config
        self.data_path = self.data_config.data_path
        self.dataset_name = self.data_config.dataset_name
        self.lbl_fraction = self.data_config.lbl_fraction
        self.iid = self.data_config.iid

        # Centerlized Config
        self.epochs = self.cent_config.epochs
        self.batch_size = self.cent_config.batch_size
        self.num_clusters = self.cent_config.num_clusters

        self.optimizer = self.client_config.optimizer
        self.optim_config = self.client_config.optim_config
        self.criterion = self.client_config.criterion

        self.ckpt_path = self.checkpoint_config.ckpt_path
        self.ckpt_save_freq = self.checkpoint_config.ckpt_save_freq

        self.exp_dir = 'logs'
        self.exp_name = f'e-{self.dataset_name}_{self.iid}_{self.lbl_fraction}_{self.num_clusters}'
        self.exp_version = f'{0}.{0}.{1}'

    def setup(self, **init_kwargs):
        train_dataset, truth_dataset, test_dataset = create_centerlized_datasets(data_config=self.configs.data_config)
        
        print('local_datasets: ', train_dataset[0].shape, train_dataset[1].shape)
        print('truth_dataset: ', truth_dataset[0].shape, truth_dataset[1].shape)
        print('test_dataset: ', test_dataset.data.shape, test_dataset.targets.shape)

        self.model = create_model(self.model_config)

        # Annotate Unlabeled data
        train_data, accuracy, homogeneity_score = annotate_data(data=train_dataset, truth_data=truth_dataset, num_clusters=self.num_clusters, dataset_name=self.dataset_name)

        # prepare hold-out dataset for evaluation
        transform = torchvision.transforms.ToTensor()
        if self.dataset_name in ['CIFAR10', 'CIFAR100', 'BelgiumTSC']:
            transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_data = CustomTensorDataset((train_data[0], train_data[1]), transform=transform)
        self.test_data = CustomTensorDataset((torch.Tensor(test_dataset.data), torch.Tensor(test_dataset.targets)), transform=transform)
        
        self.data_module = DataModule(self.data_config, self.train_data, self.test_data, num_classes= self.model_config.num_classes, batch_size=self.batch_size, num_workers= self.batch_size)
        self._model = LitModel(self.model, learning_rate=2e-4)
        
        self.checkpointing = ModelCheckpoint(dirpath=self.exp_dir + '/checkpoints', monitor='val_acc', mode="max")
        self.es = EarlyStopping(monitor='val_acc', min_delta=0.00, patience=20, verbose=False, mode="max")
        self.mt = MetricTracker()

        self.tf_logger = TensorBoardLogger(save_dir=f'{self.exp_dir}/tb/cent/{self.dataset_name}/{self.iid}/{self.lbl_fraction}/{self.num_clusters}', name=self.exp_name, version=self.exp_version)
        self.tf_logger.log_hyperparams({"dataset": self.dataset_name, "iid": self.iid, "lbl_fraction": self.lbl_fraction, "clusters": self.num_clusters}, {'val_acc': 0})
        self.tf_logger.log_metrics({'Data/accuracy': accuracy}, 0)
        self.tf_logger.log_metrics({'Data/homogeneity_score': homogeneity_score}, 0)
        
        self.trainer = pl.Trainer(
            num_nodes=1,
            accelerator="gpu",
            devices=2, 
            strategy= DDPStrategy(find_unused_parameters=False), # 'ddp'
            max_epochs=self.epochs,
            callbacks=[self.mt],
            logger=[self.tf_logger]
        )

    def train_model(self):
        self.trainer.fit(self._model, self.data_module)
        self.tf_logger.log_metrics({'Max/accuracy': max(self.mt.val_accs)}, 0)
        
    def evaluate_model(self):
        self.trainer.test(self._model, self.data_module)

    def fit(self):
        self.train_model()
        self.evaluate_model()
