#!/bin/python

from utils.train import Trainer # Default custom training class
from models.resnet import *
from torchvision import models

# GPU Check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:  " + str(device))

# Use different train/test data augmentations
transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Get Cifar 10 Datasets
save='./data/Cifar10'
transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(20),
        transforms.RandomCrop(32, (2, 2), pad_if_needed=False, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Get Cifar 10 Datasets
trainset = torchvision.datasets.CIFAR10(root=save, train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=save, train=False, download=True, transform=transform_test)

# Get Cifar 10 Dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=4)

#################################
# Load the pre-trained model
#################################

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 10)
)


model_ft = model_ft.to(device)

# Loss function
cost = nn.CrossEntropyLoss()

# Optimizer
lr = 0.0005
# opt = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
opt = torch.optim.Adam(model_ft.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-4) #0.0005 l2_factor.item()

# Create a trainer
trainer = Trainer(model_ft, opt, cost, name="Transfer-learning",lr=lr , use_lr_schedule=True, device=device)

# Run training
epochs = 25
trainer.Train(trainloader, epochs, testloader=testloader)
# trainer.Train(trainloader, epochs) # check train error

print('done')
