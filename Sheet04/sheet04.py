import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '  '

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class ShallowModel(nn.Module):
   def __init__(self):
       super(ShallowModel, self).__init__()
       self.conv_layers = nn.Sequential(
           nn.Conv2d(1, 10, kernel_size=3,stride=1, padding=0),
           nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=0),
           nn.MaxPool2d(kernel_size=2,stride=2),

       )
       
       self.fc = nn.Linear(20*12*12, 10)
       
       
   def forward(self,x):
       x = self.conv_layers(x)
       x = x.view(x.size(0),-1)
       return self.fc(x)
       

class WiderModel(nn.Module):
   def __init__(self):
       super(WiderModel, self).__init__()
       self.conv1 = nn.Conv2d(1, 10, kernel_size=3,stride=1, padding=0)
       self.conv2 = nn.Conv2d(1, 10, kernel_size=3,stride=1, padding=0)
       self.conv3 = nn.Conv2d(1, 10, kernel_size=3,stride=1, padding=0)
       self.conv4 = nn.Conv2d(1, 10, kernel_size=3,stride=1, padding=0)
       
       self.max_pool =  nn.MaxPool2d(kernel_size=2,stride=2)
       
       self.fc = nn.Linear(10*4*13*13, 10)
       
       
   def forward(self,x):
       x1 = self.conv1(x)
       x2 = self.conv2(x)
       x3 = self.conv3(x)
       x4 = self.conv4(x)
       concatenated = torch.cat([x1,x2,x3,x4], dim=1)
       pooled = self.max_pool(concatenated)
       
       flattened= pooled.view(pooled.size(0),-1)
      
       return self.fc(flattened)


class DeeperModel(nn.Module):
   def __init__(self, batchNorm):
       super(DeeperModel, self).__init__()
       if (not batchNorm):
           self.conv_layers = nn.Sequential(
               nn.Conv2d(1, 10, kernel_size=3,stride=1, padding=0),
               nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=0),
               nn.MaxPool2d(kernel_size=2,stride=2),
               nn.Conv2d(20, 40, kernel_size=3,stride=1, padding=0),
               nn.Conv2d(40, 80, kernel_size=3, stride=1, padding=0),
               nn.MaxPool2d(kernel_size=2,stride=2),
           )
       else:
           self.conv_layers = nn.Sequential(
               nn.Conv2d(1, 10, kernel_size=3,stride=1, padding=0),
                      torch.nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
               nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=0),
                                     torch.nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
               nn.MaxPool2d(kernel_size=2,stride=2),
               nn.Conv2d(20, 40, kernel_size=3,stride=1, padding=0),
                                     torch.nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
               nn.Conv2d(40, 80, kernel_size=3, stride=1, padding=0),
                                     torch.nn.BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
               nn.MaxPool2d(kernel_size=2,stride=2),
           )
       
       self.fully_connected_layers = nn.Sequential(
           nn.Linear(80*4*4, 200),
           nn.Linear(200, 10)
           )

       
   def forward(self,x):
       x = self.conv_layers(x)
       x = x.view(x.size(0),-1)
       return self.fully_connected_layers(x)

def main():
    ## Get data
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    ## Training Parameters
    nepochs = 10
    batch_size = 64

    ## Create Dataloaders
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size)


    ### Create Model
    #model = ShallowModel()

    #model = WiderModel()
    
    #model = DeeperModel(batchNorm=False)
    model = DeeperModel(batchNorm=True)

    ### Define Opitmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # ### Train
    
    for epoch in range(nepochs):
        criterion  = torch.nn.CrossEntropyLoss()
        avg_epoch_loss = 0
        print("Starting Epoch {}".format(epoch+1))
        for data,labels in train_loader:
            optimizer.zero_grad()
            predictions = model(data)  
            loss = criterion (predictions, labels)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            print("current batch loss: {}".format(batch_loss))
            avg_epoch_loss+=batch_loss
        avg_epoch_loss/= len(train_loader)
        print("Average loss this epoch: {}".format(avg_epoch_loss))
            
            
        

    ### Save Model for sharing
    torch.save(model.state_dict(), './model')

    ## Test
    total_right = 0
    for data, labels in test_loader:
        predictions = model(data)
        predicted_classes = torch.argmax(predictions, dim=1)
        predicted_correct = torch.sum(labels == predicted_classes)
        print(predicted_classes)
        print(labels)
        print(predicted_correct)
        total_right+=predicted_correct
    acc = total_right / (len(test_loader)*batch_size)
    print("Model accuracy on test set is: {}".format(acc))



if __name__ == '__main__':
    main()
