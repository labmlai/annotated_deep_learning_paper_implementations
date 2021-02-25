#!/bin/python

# Custom classes
from models.mlp import MLP
from utils.train import Trainer
from models.resnet import *

# GPU Check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:  " + str(device))

#Use different train/test data augmentations
transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(20),
        transforms.RandomCrop(32, (2, 2), pad_if_needed=False, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Get Cifar 10 Datasets
save='./data/Cifar10'
trainset = torchvision.datasets.CIFAR10(root=save, train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=save, train=False, download=True, transform=transform_test)

# Get Cifar 10 Dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, 
                                         shuffle=False, num_workers=4)

epochs = 50

#################################
# Create the assignment Resnet (part a)
#################################
def MyResNet():
    resnet = ResNet(in_features= [32, 32, 3],
                    num_class=10,
                    feature_channel_list = [128, 256, 512],
                    batch_norm= True,
                    num_stacks=1
                    )

    # Create MLP
    # Calculate the input shape
    s = resnet.GetCurShape()
    in_features = s[0]*s[1]*s[2]

    mlp = MLP(in_features,
                 10,
                 [], #512, 1024, 512
                 [],
                 use_batch_norm=False,
                 use_dropout=False,
                 use_softmax=False,
                 device=device)

    resnet.AddMLP(mlp)
    return resnet

model = MyResNet()
model.to(device=device)
summary(model, (3, 32,32))

# Optimizer
opt = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.95), weight_decay=1e-8) #0.0005 l2_factor.item()

# Loss function
cost = nn.CrossEntropyLoss()

# Create a trainer
trainer = Trainer(model, opt, cost, name="MyResNet", device=device, use_lr_schedule =True)

# Run training
trainer.Train(trainloader, epochs, testloader=testloader)

print('done')
