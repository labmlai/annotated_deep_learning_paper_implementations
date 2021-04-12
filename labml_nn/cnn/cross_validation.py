
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torchsummary import summary
import torch.nn as nn

# from models.mlp import MLP
# from utils.utils import *
# from utils.train_dataset import *
#from nutsflow import Take, Consume
#from nutsml import *
from utils.dataloader import *
from models.cnn import CNN
from utils.train import Trainer

from utils.cv_train import *

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:  " + str(device))

# Cifar 10 Datasets location
save='./data/Cifar10'

# Transformations train
transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load train dataset and dataloader
trainset = LoadCifar10DatasetTrain(save, transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=4)

# Transformations test (for inference later)
transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load test dataset and dataloader (for inference later)
testset = LoadCifar10DatasetTest(save, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=4)

# Specify loss function
cost = nn.CrossEntropyLoss()

epochs=25  #10
splits = 4 #5

# Training - Cross-validation
history = cross_val_train(cost, trainset, epochs, splits, device=device)

# Inference
best_model, best_val_accuracy = retreive_best_trial()
print("Best Validation Accuracy = %.3f"%(best_val_accuracy))

# Testing
accuracy = Test(best_model, cost, testloader, device=device)
print("Test Accuracy = %.3f"%(accuracy['val_acc']))
