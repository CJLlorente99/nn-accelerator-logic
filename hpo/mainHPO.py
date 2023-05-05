import random

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import os
import torch.nn as nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from hpo.auxFunction import train
from hpo.binaryNNHPO import BinaryNeuralNetwork


def main(num_samples=10, max_num_epochs=10):

    # Check mps maybe if working in MacOS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {'hiddenLayers': tune.randint(3, 4),
              'npl': tune.choice([4096]),
              'lr': tune.uniform(3e-3, 3e-6),
              'wd': tune.uniform(0, 1e-6)}

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["npl", 'hiddenLayers'],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(train),
        config=config,
        resources_per_trial={"cpu": 6},
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    main(num_samples=20, max_num_epochs=20)

