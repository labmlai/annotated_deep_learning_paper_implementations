#!/bin/python

import numpy as np
import os
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from utils.train import Trainer
from models.cnn import GetCNN

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:  " + str(device))

#
num_samples= 40  # for multiple trials
max_num_epochs= 25
gpus_per_trial= 1

# Cifar 10 Datasets location
data_dir = './data/Cifar10'

"""
Code has been referenced from the official ray tune documentation
ASHA
https://docs.ray.io/en/master/tune/api_docs/schedulers.html#tune-scheduler-hyperband

PBT
https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-scheduler-pbt
"""

"""config - returns a dict of hyperparameters

Selecting different hyperparameters for tuning
    l1 : Number of units in first fully connected layer
    l2 : Number of units in second fully connected layer
    lr : Learning rate
    decay : Decay rate for regularization
    batch_size : Batch size of test and train data
"""
config = {
    "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)), # eg. 4, 8, 16 .. 512
    "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)), # eg. 4, 8, 16 .. 512
    "lr": tune.loguniform(1e-4, 1e-1), # Sampling from log uniform distribution
    "decay": tune.sample_from(lambda _: 10 ** np.random.randint(-7, -3)), # eg. 1e-7, 1e-6, .. 1e-3
    "batch_size": tune.choice([32, 64, 128, 256])
}

# calling trainer
trainer = Trainer(device=device)

"""ASHA (Asynchronous Successive Halving Algorithm) scheduler
        max_t              : Maximum number of units per trail (can be time or epochs)
        grace_period       : Stop trials after specific number of unit if model is not performing well (can be time or epochs)
        reduction_factor   : Set halving rate
"""
scheduler = ASHAScheduler(
    max_t=max_num_epochs,
    grace_period=4,
    reduction_factor=4)



"""Population based training scheduler
    time_attr             : Can be time or epochs
    metric                : Objective of training (loss or accuracy)
    perturbation_interval : Perturbation occur after specified unit (can be time or epochs)
    hyperparam_mutations  : Hyperparameters to mutate
"""
scheduler = PopulationBasedTraining(
        time_attr= "training_iteration", # epochs
        metric='loss', # loss is objective function
        mode='min', # minimizing loss is objective of training
        perturbation_interval=5.0, # after 5 epochs perturbate
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-4, 1e-5], # choose from given learning rates
            "batch_size": [64, 128, 256], # choose from given batch sizes
            "decay": tune.uniform(10**-8, 10**-4) # sample from uniform distribution
            }
        )

result = tune.run(
    tune.with_parameters(trainer.Train_ray, data_dir=data_dir),
    name="ray_test_basic-CNN", # name for identifying models (checkpoints)
    scheduler=scheduler, # select scheduler PBT or ASHA
    resources_per_trial={"cpu": 8, "gpu": gpus_per_trial}, # select number of CPUs or GPUs
    config=config, # input config dict consisting of different hyperparameters
    stop={
        "training_iteration": max_num_epochs, # stopping criterea
    },
    metric="loss", # uncomment for ASHA scheduler
    mode="min", # uncomment for ASHA scheduler
    num_samples=num_samples,
    verbose=True, # keep to true to check how training progresses
    fail_fast=True, # fail on first error
    keep_checkpoints_num=5, # number of checkpoints to be saved per num_samples

)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best configuration: {}".format(best_trial.config))
print("Best validation loss: {}".format(best_trial.last_result["loss"]))
print("Best validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))


best_trained_model = GetCNN(best_trial.config["l1"], best_trial.config["l2"])
best_trained_model.to(device)
checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
model_state, optimizer_state = torch.load(checkpoint_path)
best_trained_model.load_state_dict(model_state)

# Check accuracy of best model
test_acc =  trainer.Test(best_trained_model, save=data_dir)
print("Best Test accuracy: {}".format(test_acc))