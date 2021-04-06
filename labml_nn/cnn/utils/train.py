#!/bin/python

import torch.nn as nn
import matplotlib.pyplot as plt
import os
from models.cnn import GetCNN
from ray import tune
from utils.dataloader import * # Get the transforms


class Trainer():
    def __init__(self, name="default", device=None):
        self.device = device

        self.epoch = 0
        self.start_epoch = 0
        self.name = name

    # Train function
    def Train(self, net, trainloader, testloader, cost, opt, epochs = 25):

        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

        # Optimizer and Cost function
        self.opt = opt
        self.cost = cost

        # Bookkeeping
        train_loss = torch.zeros(epochs)
        self.epoch = 0
        train_steps = 0
        accuracy = torch.zeros(epochs)

        # Training loop
        for epoch in range(self.start_epoch, self.start_epoch+epochs):
            self.epoch = epoch+1
            self.net.train()  # Enable Dropout

            # Iterating over train data
            for data in self.trainloader:
                if self.device:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    images, labels = data[0], data[1]

                self.opt.zero_grad()

                # Forward + backward + optimize
                outputs = self.net(images)
                epoch_loss = self.cost(outputs, labels)
                epoch_loss.backward()
                self.opt.step()
                train_steps+=1

                train_loss[epoch] += epoch_loss.item()
            loss_train = train_loss[epoch] / train_steps

            accuracy[epoch] = self.Test() #correct / total

            print("Epoch %d LR %.6f Train Loss: %.3f Test Accuracy: %.3f" % (
            self.epoch, self.opt.param_groups[0]['lr'], loss_train, accuracy[epoch]))

            # Save best model
            if accuracy[epoch] >= torch.max(accuracy):
                self.save_best_model({
                    'epoch': self.epoch,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.opt.state_dict(),
                })

        self.plot_accuracy(accuracy)

    # Test over testloader loop
    def Test(self, net = None, save=None):
        # Initialize dataloader
        if save == None:
            testloader = self.testloader
        else:
            testloader = Dataloader_test(save, batch_size=128)

        # Initialize net
        if net == None:
            net = self.net

        # Disable Dropout
        net.eval()

        # Bookkeeping
        correct = 0.0
        total = 0.0

        # Infer the model
        with torch.no_grad():
            for data in testloader:
                if self.device:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    images, labels = data[0], data[1]

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # compute the final accuracy
        accuracy = correct / total
        return accuracy

    # Train function modified for ray schedulers
    def Train_ray(self, config, checkpoint_dir=None, data_dir=None):
        epochs = 25

        self.net = GetCNN(config["l1"], config["l2"])
        self.net.to(self.device)

        trainloader, valloader = Dataloader_train_valid(data_dir, batch_size=config["batch_size"])

        # Optimizer and Cost function
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=config["decay"])
        self.cost = nn.CrossEntropyLoss()

        # restoring checkpoint
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            # checkpoint = checkpoint_dir
            model_state, optimizer_state = torch.load(checkpoint)
            self.net.load_state_dict(model_state)
            self.opt.load_state_dict(optimizer_state)

        self.net.train()

        # Record loss/accuracies
        train_loss = torch.zeros(epochs)
        self.epoch = 0
        train_steps = 0
        for epoch in range(self.start_epoch, self.start_epoch+epochs):
            self.epoch = epoch+1

            self.net.train()  # Enable Dropout
            for data in trainloader:
                # Get the inputs; data is a list of [inputs, labels]
                if self.device:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    images, labels = data[0], data[1]

                self.opt.zero_grad()
                # Forward + backward + optimize
                outputs = self.net(images)
                epoch_loss = self.cost(outputs, labels)
                epoch_loss.backward()
                self.opt.step()
                train_steps+=1

                train_loss[epoch] += epoch_loss.item()

            # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            self.net.eval()
            for data in valloader:
                with torch.no_grad():
                    # Get the inputs; data is a list of [inputs, labels]
                    if self.device:
                        images, labels = data[0].to(self.device), data[1].to(self.device)
                    else:
                        images, labels = data[0], data[1]

                    # Forward + backward + optimize
                    outputs = self.net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = self.cost(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            # Save checkpoints
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (self.net.state_dict(), self.opt.state_dict()), path)

            tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

        return train_loss

    def plot_accuracy(self, accuracy, criterea = "accuracy"):
        plt.plot(accuracy.numpy())
        plt.ylabel("Accuracy" if criterea == "accuracy" else "Loss")
        plt.xlabel("Epochs")
        plt.show()


    def save_checkpoint(self, state):
        directory = os.path.dirname("./save/%s-checkpoints/"%(self.name))
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(state, "%s/model_epoch_%s.pt" %(directory, self.epoch))

    def save_best_model(self, state):
        directory = os.path.dirname("./save/%s-best-model/"%(self.name))
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(state, "%s/model.pt" %(directory))