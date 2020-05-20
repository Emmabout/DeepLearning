import torch
import math
import matplotlib.pyplot as plt
torch.set_grad_enabled(False)
nb_epochs = 25

def generate_set():
    training_set = torch.empty(1000,2).uniform_(0,1)  #x et y
    training_classes = torch.empty(1000)
    testing_set = torch.empty(1000,2).uniform_(0,1)  #x et y
    testing_classes = torch.empty(1000)

    r = torch.empty(1,1).fill_(1/(2*math.pi)).pow(1/2)

    for i in range (1000):
        if ((training_set[i] - torch.Tensor([0.5,0.5])).pow(2).sum()).pow(1/2).item() < r.item():
            training_classes[i] = 1
        else:
            training_classes[i] = 0

        if ((testing_set[i] - torch.Tensor([0.5,0.5])).pow(2).sum()).pow(1/2).item() < r.item():
            testing_classes[i] = 1
        else:
            testing_classes[i] = 0
    return training_set, training_classes, testing_set, testing_classes

#LINEAR MODULE (FULLY CONNECTED LAYERS)
class Linear(object):
    def __init__(self, nb_data_in, nb_data_out):
        k = math.sqrt(1/nb_data_in)
        self.weight = torch.empty(nb_data_out,nb_data_in).uniform_(-k,k)
        self.bias = torch.empty(nb_data_out,1).uniform_(-k,k)
        self.grad_weight = None
        self.grad_bias = None
        self.input = None
        
    def updateparam(self, lr):
        for i in range(len(self.grad_weight)): 
            self.weight -= lr * self.grad_weight
            self.bias -= lr * self.grad_bias
    
    def forward(self , input):
        self.input = input
        y_correct_dim = ((self.weight).matmul(input.t())+self.bias)
        return y_correct_dim.t()
    
    def backward(self, gradwrtoutput):
        gradaccumulated = gradwrtoutput.matmul(self.weight)
        self.grad_bias = gradwrtoutput.t()
        self.grad_weight = gradwrtoutput.t().matmul(self.input)
        return gradaccumulated
        
    def param(self):
        output = [[self.weight, self.grad_weight], [self.bias, self.grad_bias]]
        return output

#RELU MODULE
class ReLU():
    def __init__(self):
        self.input = None
    
    def forward(self, input):
        self.input = input
        return torch.max(input,torch.zeros_like(input))
        
    def backward(self, gradwrtoutput): 
        dx = ((self.input)>=0).float()
        return dx*gradwrtoutput 

    def param(self): 
        return [] #no parameters

#TANH MODULE
class Tanh():
    def __init__(self):
        self.input = None
        
    def forward(self, input):
        self.input = input
        return torch.tanh(input)
    
    def backward(self, gradwrtoutput):
        return (1 - torch.tanh(self.input).pow(2))*gradwrtoutput 

    def param(self):
        return [] #No parameters

#LOSSMSE MODULE
class LossMSE():
    
    def forward(self, input, target): 
        loss = torch.mean((input-target).pow(2))
        return loss
        
    def backward(self, input, target):
        target = target.unsqueeze(0)
        dloss = (2*(input - target))/(input.size(1))
        return dloss

    def param(self):
        return [] #No parameters

#Creating a [1000,2] tensor for the classes
def class_into_2(classes):
    t = torch.empty(classes.size(0),2).zero_()
    for n in range (classes.size(0)):
        t[n,int(classes[n].item())] = 1
    return t

#SEQUENTIAL MODULE
class Sequential():
    def __init__(self, *input):
        self.input = input
    
    def forward(self, x):
        for inp in self.input:
            x = inp.forward(x)
        return x
        
    def backward(self, d_loss): 
        for inp in reversed(self.input):
            d_loss = inp.backward(d_loss)
            
    def update(self,lr):
        for inp in self.input:
            if hasattr(inp, 'updateparam'):
                inp.updateparam(lr)

#TRAIN THE MODEL
def train_model(model, input, target, lr):
    loss = LossMSE()
    liste = [0]*nb_epochs
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(input.shape[0]):
            output = model.forward(input[b].unsqueeze(0))
            d_loss = loss.backward(output, target[b])
            
            model.backward(d_loss)
            model.update(lr)
        
            sum_loss += loss.forward(output, target[b])
        liste[e] = sum_loss
        print(e,sum_loss.item())
        
    plt.plot(liste)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

#COMPUTE NUMBER OF ERRORS
def compute_nb_errors(model, input, target):
    nb_errors = 0

    for j in range(input.shape[0]):
        output = model.forward(input[j].unsqueeze(0))
        pred = output.squeeze().max(0)[1].item()
        #print(output)
        if target[j, pred].item() < 0.5: 
                nb_errors = nb_errors + 1
    return nb_errors
    

##################### MODEL USING RELU

training_set, training_classes, testing_set, testing_classes = generate_set()
train_target = class_into_2(training_classes)
test_target = class_into_2(testing_classes)
print("-------------3 HIDDEN LAYERS SIZE 25, RELU-------------")
#CREATE A MODEL WITH 2 INPUT UNITS, 3 HIDDEN LAYERS OF 25 UNITS AND RELU
model = Sequential(Linear(2,25),
                   ReLU(),
                   Linear(25,25),
                   ReLU(),
                   Linear(25,25),
                   ReLU(),
                   Linear(25,25),
                   ReLU(),
                   Linear(25,2))
lr = 0.01

plt.figure()
plt.title("Evolution of Loss using 3 hidden layers of size 25 and ReLU")

train_model(model, training_set, train_target, lr)

train_errors = compute_nb_errors(model, training_set, train_target)
test_errors = compute_nb_errors(model, testing_set, test_target)

print('Train error {:0.2f}% {:0.2f}/{:d}'.format((100 * train_errors) / training_set.size(0),
                                                          train_errors, training_set.size(0)))
print('Test error {:0.2f}% {:0.2f}/{:d}'.format((100 * test_errors) / testing_set.size(0),
                                                          test_errors, testing_set.size(0)))


####################MODEL USING TANH
training_set, training_classes, testing_set, testing_classes = generate_set()
train_target = class_into_2(training_classes)
test_target = class_into_2(testing_classes)
print("-------------3 HIDDEN LAYERS SIZE 25, TANH-------------")
#CREATE A MODEL WITH 2 INPUT UNITS, 3 HIDDEN LAYERS OF 25 UNITS AND TANH
model = Sequential(Linear(2,25),
                   Tanh(),
                   Linear(25,25),
                   Tanh(),
                   Linear(25,25),
                   Tanh(),
                   Linear(25,25),
                   Tanh(),
                   Linear(25,2))
lr = 0.01

plt.figure()
plt.title("Evolution of Loss using 3 hidden layers of size 25 and Tanh")

train_model(model, training_set, train_target, lr)

train_errors = compute_nb_errors(model, training_set, train_target)
test_errors = compute_nb_errors(model, testing_set, test_target)

print('Train error {:0.2f}% {:0.2f}/{:d}'.format((100 * train_errors) / training_set.size(0),
                                                          train_errors, training_set.size(0)))
print('Test error {:0.2f}% {:0.2f}/{:d}'.format((100 * test_errors) / testing_set.size(0),
                                                          test_errors, testing_set.size(0)))

plt.show()