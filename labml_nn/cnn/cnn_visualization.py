#!/bin/python

import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from functools import partial
from skimage.filters import sobel, sobel_h, roberts
from models.cnn import CNN
from utils.dataloader import *
from utils.train import Trainer

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

# Transformations test
transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load test dataset and dataloader
testset = LoadCifar10DatasetTest(save, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=4)

# Create CNN model
def GetCNN():
    cnn = CNN( in_features=(32,32,3),
                out_features=10,
                conv_filters=[32,32,64,64],
                conv_kernel_size=[3,3,3,3],
                conv_strides=[1,1,1,1],
                conv_pad=[0,0,0,0],
                max_pool_kernels=[None, (2,2), None, (2,2)],
                max_pool_strides=[None,2,None,2],
                use_dropout=False,
                use_batch_norm=True, #False
                actv_func=["relu", "relu", "relu", "relu"],
                device=device
        )

    return cnn

model = GetCNN()

# Display model specifications
summary(model, (3,32,32))

# Send model to GPU
model.to(device)

# Specify optimizer
opt = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.95))

# Specify loss function
cost = nn.CrossEntropyLoss()

# Train the model
trainer = Trainer(device=device, name="Basic_CNN")
epochs = 5
trainer.Train(model, trainloader, testloader, cost=cost, opt=opt, epochs=epochs)

# Load best saved model for inference
model_loaded = GetCNN()

# Specify location of saved model
PATH = "./save/Basic_CNN-best-model/model.pt"
checkpoint = torch.load(PATH)

# load the saved model
model_loaded.load_state_dict(checkpoint['state_dict'])

# intialization for hooks and storing activation of ReLU layers
activation = {}
hooks = []

# Hook function saves activation of a particular layer
def hook_fn(model, input, output, name):
    activation[name] = output.cpu().detach().numpy()

# Registering hooks
count =0
conv_count = 0
for name, layer in model_loaded.named_modules():
    if isinstance(layer, nn.ReLU):
        count +=1
        hook = layer.register_forward_hook(partial(hook_fn, name=f"{layer._get_name()}-{count}")) #f"{type(layer).__name__}-{name}"
        hooks.append(hook)
    if isinstance(layer, nn.Conv2d):
        conv_count += 1

# Displaying image used for inference
data, _ = trainset[15]
imshow(data)

# Infering model to save activation of ReLU layers
output = model_loaded(data[None].to(device))

# Removing hooks
for hook in hooks:
    hook.remove()

# Function to display output of a particular ReLU layer
def output_one_layer(layer_num):
    assert 1 <= layer_num <= len(activation), "Wrong layer number"

    layer_name = f"ReLu-{layer_num}"
    act = activation[f"ReLU-{layer_num}"]
    if act.shape[1]==32:
        rows = 4
        columns = 8
    elif act.shape[1]==64:
        rows = 8
        columns = 8

    fig = plt.figure(figsize=(rows, columns))
    for idx in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, idx)
        plt.imshow(sobel(act[0][idx-1]), cmap=plt.cm.gray)

        # try different filters
        # plt.imshow(act[0][idx-1], cmap='viridis', vmin=0, vmax=act.max())
        # plt.imshow(act[0][idx - 1], cmap='hot')
        # plt.imshow(roberts(act[0][idx - 1]), cmap=plt.cm.gray)
        # plt.imshow(sobel_h(act[0][idx-1]), cmap=plt.cm.gray)

        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Function to display output of all ReLU layer after Convulution layers
def output_all_layers():
    for [name, output], count in zip(activation.items(), range(conv_count)):
        if output.shape[1] == 32:
            _, axs = plt.subplots(8, 4, figsize=(8, 4))
        elif output.shape[1] == 64:
            _, axs = plt.subplots(8, 8, figsize=(8, 8))

        for ax, out in zip(np.ravel(axs), output[0]):
            ax.imshow(sobel(out), cmap=plt.cm.gray)
            ax.axis('off')

        plt.suptitle(name)
        plt.tight_layout()
        plt.show()

# Choose either one to display
output_one_layer(layer_num=3) # choose layer number
output_all_layers()

