import time
import torch
import subprocess
from queue import Queue



class Experiment:
    def __init__(self, exps_per_dev=2):
        print('init')
        self.devices = [i for i in range(torch.cuda.device_count())]
        self.exps_per_dev = exps_per_dev
        self.device_bucket = [[None] * self.exps_per_dev for _ in range(torch.cuda.device_count())]
        self.experiments = Queue()   
        self.close_flag = False

    def add_experiment(self, exp):
        self.experiments.put(exp)

    def is_all_device_free(self):
        for dev in self.device_bucket:
            for p in dev:
                if p is not None:
                    return False
        return True

    def is_any_device_free(self):
        for dev in self.device_bucket:
            for proc in dev:
                if proc is None:
                    return True
        return False

    def get_free_device(self):
        for dev_id, dev in enumerate(self.device_bucket):
            for exp_id, p in enumerate(dev):
                if p is None:
                    return dev_id, exp_id
        return None, None

    def free_device(self):
        for i, dev in enumerate(self.device_bucket):
            for j, p in enumerate(dev):
                if p is not None and p.poll() is not None:
                    self.device_bucket[i][j] = None

    def run(self):
        while not self.experiments.empty() or self.is_all_device_free() or not self.close_flag:
            self.free_device()
            time.sleep(1)

            if self.is_any_device_free() and not self.experiments.empty():
                dev_id, exp_id = self.get_free_device()
                
                exp = self.experiments.get() + str(dev_id)                   
                
                self.device_bucket[dev_id][exp_id] = subprocess.Popen(exp, shell=True)
    
    def close(self):
        self.close_flag = True