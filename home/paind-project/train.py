#imports
import torch
from torch import nn
from torch import optim
from torchvision import transforms, models, datasets 

import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

import json

from PIL import Image

with open(json1, 'r') as f:
    cat_to_name = json.load(f)

    
if gpu == 'yes':
    device = 'cuda:0'
else:
    device = 'cpu'
device = device

#directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#transforms
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225]),
                                     transforms.Normalize([0.5, 0.5, 0.5],
                                                          [0.5, 0.5, 0.5])]
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225]),
                                     transforms.Normalize([0.5, 0.5, 0.5],
                                                          [0.5, 0.5, 0.5])])
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225]),
                                     transforms.Normalize([0.5, 0.5, 0.5],
                                                          [0.5, 0.5, 0.5])])
#datasets
train_dataset = datasets.ImageFolder(train_dir, transform = data_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = data_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = data_transforms) 

#loaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

#creating neural network
class Network(nn.Module): 
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):     
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])     # creates a linear network from input to the first hidden layer
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:]) # creates a tuple between every connection of hidden layers
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])  # makes each of those connections into a linear network
        
        self.output = nn.Linear(hidden_layers[-1], output_size) # creates a linear network from last hidden layer to output
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)_

model_input = input("Do you want to make your own network('1'), have a densenet neural network('2'), or have a resnet neural network('3')")
if model_input = '1':
    #asking user input
    hidden_layer_number = int(input("How many hidden layers do you want?"))
    hidden_layer_list = []
    for i in range(hidden_layer_number):
        y = "How large will hidden layer #" + i + "be?"
        x = int(input(y))
        x.append(hidden_layer_list)
                                     
    model = Network(50176, 102, hidden_layer_list, drop_p=0.5) # creating the neural network model and defining parameters (input, hidden layers, output, probability)          
                                     
    criterion = nn.NLLLoss()                             # use negative log los as criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
else if model_input = '2':
    model = models.densenet121(pretrained = True)
                                     
    criterion = nn.NLLLoss()                             # use negative log los as criterion
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
else if model_input = '3':
    model = models.resnet18(pretrained = True)
                                     
    criterion = nn.NLLLoss()                             # use negative log los as criterion
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model = model

if model_input = '2' or model_input = '3':
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 500)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(500, 102)),
                                ('output', nn.LogSoftmax(dim = 1))]))
    model.classifier = classifier
                                     
                                     
if gpu and torch.cuda.is_available():
      model.cuda()

for param in model.parameters():
    param.requires_grad = True
    
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to("cuda"), labels.to("cuda")
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean() 

    return test_loss, accuracy

epoch_number = input("How many epochs do you want the neural network to run through?")

model.to(device)
epochs = epoch_number
print_every = 5 
steps = 0
for e in range(epochs):

    # puts it in training mode so that dropout is enabled
    model.train()
    
    for images, labels in trainloader:
        steps += 1
        
        images, labels = images.to(device), labels.to(device) 
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference and removes dropout
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, trainloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(trainloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(trainloader)))
            
            running_loss = 0
            
            # Make sure training is back on and dropout is enabled once more
            model.train()
            
correct = 0 
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to("cuda"), labels.to("cuda")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

torch.save({'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx,
            'epoch': epochs,
            'optimizer' : optimizer.state_dict(),
            'Learning_Rate' : 0.005})
