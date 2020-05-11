import torch
from torch import nn
from torch.nn import functional as F
print_shapes_Net = False
#######################NETWORKS###########################
#Shallow network
class Shallow_Net(nn.Module):
    def __init__(self, nb_hidden):
        super(Shallow_Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)) #6x6
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)) #conv : 4x4, maxpool : 2x2
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

#Siamese network
class Siamese_net(nn.Module):
    def __init__(self):
        super(Siamese_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 10)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, data):
        if print_shapes_Net:
            print("initial", data.shape) #100 2 14 14
            
        final_layer = []
        for i in range(2):
            x = data[:,i,:,:]
            len0 = x.shape[0]
            x = torch.reshape(x, (len0, 1, 14, 14))
            if print_shapes_Net:
                print("X START",x.shape) 
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
            if print_shapes_Net:
                print("conv1",x.shape) 
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            if print_shapes_Net:
                print("conv2",x.shape)
            x = F.relu(self.fc1(x.view(-1, 256)))
            if print_shapes_Net:
                print("fc1",x.shape) 
                
            final_layer.append(x)
            
        final_layer = torch.cat((final_layer[0], final_layer[1]), 1)
                
        final_layer = self.fc2(final_layer)
        if print_shapes_Net:
            print("final",final_layer.shape) 
            
        return final_layer

#Siamese + auxiliary loss
class Siamese_net_auxiliary(nn.Module):
    def __init__(self, nb_hidden):
        super(Siamese_net_auxiliary, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, data):
        if print_shapes_Net:
            print("initial", data.shape) #100 2 14 14
            
        class_layer = []
        final_layer = []
        for i in range(2):
            x = data[:,i,:,:]
            len0 = x.shape[0]
            x = torch.reshape(x, (len0, 1, 14, 14))
            
            if print_shapes_Net:
                print("X START",x.shape) #[100, 1, 14, 14]
            
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
            if print_shapes_Net:
                print("conv1",x.shape) #[100, 32, 6, 6]
                
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            if print_shapes_Net:
                print("conv2",x.shape) #[100, 64, 2, 2]
            
            x = F.relu(self.fc1(x.view(-1, 256)))
            if print_shapes_Net:
                print("fc1",x.shape) #[100, 64]
            
            x = F.relu(self.fc2(x))
            if print_shapes_Net:
                print("fc2",x.shape) #[100 10]
                
            final_layer.append(x)
            class_layer.append(x)
            #class_layer.append(x.reshape(x.shape[0], 1, 10))
            
        final_layer = torch.cat((final_layer[0], final_layer[1]), 1)
        class_layer = torch.cat((class_layer[0], class_layer[1]), 1)
        class_layer = torch.reshape(class_layer, (len0, 2, 10))
        
        if print_shapes_Net:
                print("class layer",class_layer.shape) #[100, 2, 10]
                
        final_layer = self.fc3(final_layer)
        if print_shapes_Net:
            print("final",final_layer.shape) #[100 2]
            
        return class_layer, final_layer