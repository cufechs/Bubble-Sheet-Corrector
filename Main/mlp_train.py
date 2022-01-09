#!/usr/bin/env python

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from skimage.morphology import binary_dilation, binary_erosion
from skimage.transform import resize

from six.moves import urllib
from torchvision import datasets
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_1 = 512
        hidden_2 = 32
        
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.reshape(-1, 28 * 28)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x



def load_ds():
    
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    ############################################################

    num_workers = 0
    # how many samples per batch to load
    batch_size = 20


    # convert data to torch.FloatTensor
    transform = transforms.Compose([transforms.RandomRotation(degrees=25),
                                    transforms.ToTensor()
                                    ])

    # choose the training and test datasets
    train_data = datasets.MNIST(root='data', train=True,
                                       download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False,
                                      download=True, transform=transform)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)
        
    return(train_loader,test_loader)
    


def train(n_epochs=30):

    train_loader = load_ds()[0]

    model = Net()
    criterion = nn.CrossEntropyLoss()


    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    # number of epochs to train the model
    #n_epochs = 30  # suggest training between 20-50 epochs

    model.train() # prep model for training

    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
            
        # print training statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1, 
            train_loss
            ))

    torch.save(model.state_dict(), 'model_weights.pt')
    return model



def evaluate(model):

    test_loader = load_ds()[1]
    criterion = nn.CrossEntropyLoss()
    batch_size = 20
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # prep model for *evaluation*

    for data, target in test_loader:
        # forward pass
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))



def detect_digit(img, plot=False):
       
    im = img
    im = resize(im,(28,28))
    im = np.uint8(im*255)
   
    if plot:
        plt.figure()
        plt.imshow(im)
    
    # Model loading
    model = Net()
    model.load_state_dict(torch.load('model_weights.pt'))
    model.eval()
    
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Grayscale(),
                                    #transforms.Resize((28,28)),
                                    transforms.ToTensor()
                                    ])
    im = transform(im)
    
    output = model(im)
    _, preds = torch.max(output, 1)
    return preds.detach().numpy()[0]
    

    

if __name__ == "__main__":
    model = train()
    evaluate(model)
