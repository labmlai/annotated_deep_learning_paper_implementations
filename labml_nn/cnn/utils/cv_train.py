#!/bin/python

import torch

from torch.utils.data import Subset

from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from models.cnn import GetCNN
from torchsummary import summary
import torch.optim as optim
import os

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from glob import glob



def cross_val_train(cost, trainset, epochs, splits, device=None):

    patience = 4
    history = []
    kf = KFold(n_splits=splits, shuffle=True)
    batch_size = 64
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
    directory = os.path.dirname('./save/tensorboard-%s/'%(date_time))

    if not os.path.exists(directory):
        os.mkdir(directory)

    for fold, (train_index, test_index) in enumerate(kf.split(trainset.data, trainset.targets)): #dataset required - compelete training set
        comment = f'{directory}/fold-{fold}'
        writer = SummaryWriter(log_dir=comment)

        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(test_index)
        traindata = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=2)
        valdata = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=2)

        net = GetCNN()
        net.to(device)
        if fold == 0: #Printing model detials for the first time
            summary(net, (3, 32, 32))


        # Specify optimizer
        optimizer = optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.95))
        losses = torch.zeros(epochs)
        accuracies = torch.zeros(epochs)
        min_loss = None
        count = 0
        for epoch in range(epochs):
            valid_loss = 0
            running_loss = 0.0
            epoch_loss = 0.0
            train_loss = torch.zeros(epochs)
            train_steps = 0.0

            # training steps
            net.train()  # Enable Dropout
            for i, data in enumerate(traindata, 0):
                # Get the inputs; data is a list of [inputs, labels]
                if device:
                    images, labels = data[0].to(device), data[1].to(device)
                else:
                    images, labels = data

                # Forward + backward + optimize
                outputs = net(images)
                loss = cost(outputs, labels)
                loss.backward()
                optimizer.step()
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Print loss
                running_loss += loss.item()
                epoch_loss += loss.item()
                train_loss[epoch] += loss.item()
                train_steps += 1

            loss_train = train_loss[epoch] / train_steps

            # Validation
            loss_accuracy = Test(net, cost, valdata, device)

            losses[epoch] = loss_accuracy['val_loss']
            accuracies[epoch] = loss_accuracy['val_acc']
            print("Fold %d, Epoch %d, Train Loss %.4f Validation Loss: %.4f, Validation Accuracy: %.4f" % (fold+1, epoch+1, loss_train, losses[epoch], accuracies[epoch]))

            # TensorBoard
            info = {
                "Loss/train": loss_train,
                "Loss/valid": losses[epoch],
                "Accuracy/valid": accuracies[epoch]
                }

            for tag, item in info.items():
                writer.add_scalar(tag, item, global_step=epoch)

            if min_loss == None:
                min_loss = losses[epoch]

            # Early stopping refered from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
            if losses[epoch] > min_loss:
                print("Epoch loss: %.4f, Min loss: %.4f"%(losses[epoch], min_loss))
                count += 1
                print(f'Early stopping counter: {count} out of {patience}')
                if count >= patience:
                    print(f'############### EarlyStopping ##################')
                    break

            # Saving best model
            elif losses[epoch] <= min_loss:
                count = 0
                save_best_model({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'accuracy' : accuracies[epoch]
                }, fold=fold, date_time=date_time)
                min_loss = losses[epoch]

            history.append({'val_loss': losses[epoch], 'val_acc': accuracies[epoch]})
    return history

def save_best_model(state, fold, date_time):
    directory = os.path.dirname("./save/CV_models-%s/"%(date_time))
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(state, "%s/fold-%d-model.pt" % (directory, fold))

def retreive_best_trial():
    PATH = "./save/"
    best_model = GetCNN()

    content = os.listdir(PATH)
    latest_time = 0
    for item in content:
        if 'CV_models' in item:
            foldername = os.path.join(PATH, item)
            tm = os.path.getmtime(foldername)
            if tm > latest_time:
                latest_folder = foldername

    file_type = '/*.pt'
    files = glob(latest_folder + file_type)

    accuracy = 0
    for model_file in files:
        checkpoint = torch.load(model_file)
        if checkpoint['accuracy'] > accuracy:
            best_model.load_state_dict(checkpoint['state_dict'])
            best_val_accuracy = checkpoint['accuracy']
            # Test(best_model,)

    return best_model, best_val_accuracy

def val_step(net, cost, images, labels):
    # forward pass
    output = net(images)
    # loss in batch
    loss = cost(output, labels)

    # update validation loss
    _, preds = torch.max(output, dim=1)
    acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
    acc_output = {'val_loss': loss.detach(), 'val_acc': acc}
    return acc_output

# Test over testloader/valloader loop
def Test(net, cost, testloader, device):
    # Disable Dropout
    net.eval()

    # Bookkeeping
    correct = 0.0
    total = 0.0
    loss = 0.0
    train_steps = 0.0

    # Infer the model
    with torch.no_grad():
        for data in testloader:
            if device:
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data[0], data[1]

            outputs = net(images)
            # loss in batch
            loss += cost(outputs, labels)
            train_steps+=1
            # losses[epoch] += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        loss = loss/train_steps

    accuracy = correct / total
    loss_accuracy = {'val_loss': loss, 'val_acc': accuracy} #accuracy
    return loss_accuracy