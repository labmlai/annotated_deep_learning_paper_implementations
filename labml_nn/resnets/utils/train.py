

import torch
from torch.utils.data import DataLoader, ConcatDataset
# from sklearn.model_selection import KFold
# from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
from pylab import *
import os

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR



class Trainer():
    def __init__(self, net, opt, cost, name="default", lr=0.0005, use_lr_schedule =False , device=None):
        self.net = net
        self.opt = opt
        self.cost = cost
        self.device = device
        self.epoch = 0
        self.start_epoch = 0
        self.name = name

        self.lr = lr
        self.use_lr_schedule = use_lr_schedule
        if self.use_lr_schedule:
            self.scheduler = ReduceLROnPlateau( self.opt, 'max', factor=0.1, patience=5, threshold=0.00001, verbose=True)
            # self.scheduler = StepLR(self.opt, step_size=15, gamma=0.1)

    # Train loop over epochs. Optinal use testloader to return test accuracy after each epoch
    def Train(self, trainloader, epochs, testloader=None):
        # Enable Dropout

        # Record loss/accuracies
        loss = torch.zeros(epochs)
        self.epoch = 0

        # If testloader is used, loss will be the accuracy
        for epoch in range(self.start_epoch, self.start_epoch+epochs):
            self.epoch = epoch+1

            self.net.train()  # Enable Dropout
            for data in trainloader:
                # Get the inputs; data is a list of [inputs, labels]
                if self.device:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    images, labels = data

                self.opt.zero_grad()
                # Forward + backward + optimize
                outputs = self.net(images)
                epoch_loss = self.cost(outputs, labels)
                epoch_loss.backward()
                self.opt.step()

                loss[epoch] += epoch_loss.item()

            if testloader:
                loss[epoch] = self.Test(testloader)
            else:
                loss[epoch] /= len(trainloader)

            print("Epoch %d Learning rate %.6f %s: %.3f" % (
            self.epoch, self.opt.param_groups[0]['lr'], "Accuracy" if testloader else "Loss", loss[epoch]))

            #learning rate scheduler
            if self.use_lr_schedule:
                self.scheduler.step(loss[epoch])
                # self.scheduler.step()

            # Saving best model
            if loss[epoch] >= torch.max(loss):
                self.save_best_model({
                    'epoch': self.epoch,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.opt.state_dict(),
                })

        return loss

    # Testing
    def Test(self, testloader, ret="accuracy"):
        # Disable Dropout
        self.net.eval()

        # Track correct and total
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in testloader:
                if self.device:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    images, labels = data

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def save_best_model(self, state):
        directory = os.path.dirname("./save/%s-best-model/"%(self.name))
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(state, "%s/model.pt" %(directory))

    def save_checkpoint(self, state):
        directory = os.path.dirname("./save/%s-checkpoints/"%(self.name))
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(state, "%s/model_epoch_%s.pt" %(directory, self.epoch))
        # torch.save(state, "./save/checkpoints/model_epoch_%s.pt" % (self.epoch))
