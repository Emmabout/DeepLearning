import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

mini_batch_size = 100
N = 1000 #number of pairs
nb_epoch = 25
iterations = 1

##############################TRAINING######################
def train_model(model, train_input, train_target, L, value, lr, loss):
    if loss == "MSE":
        criterion = nn.MSELoss()
    elif loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
        
    loss_plot = [0]*25
    optimizer = optim.Adam(model.parameters(), lr)
        
    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))            
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            
            if L == 1:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for p in model.parameters():
                        p.sub_(p.sign() * p.abs().clamp(max = value))
                sum_loss = sum_loss + loss.item()
            elif L==2:
                for p in model.parameters():
                    sum_loss += value * p.pow(2).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                model.zero_grad()
                loss.backward()     
                optimizer.step()
                sum_loss = sum_loss + loss.item()
                
        loss_plot[e] = sum_loss
    
    return loss_plot

def train_model_aux(model, train_input, train_classes, train_target, lr, L, value):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    loss_plot = [0]*nb_epoch
    
    for e in range(nb_epoch):
        sum_loss = 0
        sum_loss_target = 0
        sum_loss_class = 0
        for b in range(0, train_input.size(0), mini_batch_size):
          
            output_class, output_target = model(train_input.narrow(0, b, mini_batch_size))
            output_class = torch.reshape(output_class, (-1, 10))
            train_classes_reshaped = train_classes.narrow(0, b, mini_batch_size).view(-1) #with CrossEntropyLoss

            loss_class = criterion(output_class, train_classes_reshaped)
            loss_target = criterion(output_target, train_target.narrow(0, b, mini_batch_size))
            loss = loss_class + loss_target
            
            if L == 1:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for p in model.parameters():
                        p.sub_(p.sign() * p.abs().clamp(max = value))
                sum_loss = sum_loss + loss.item()
            elif L==2:
                for p in model.parameters():
                    sum_loss += value * p.pow(2).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                model.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss_class + loss_target
                sum_loss = sum_loss + loss.item()
                sum_loss_target += loss_target
                sum_loss_class += loss_class
                sum_loss = sum_loss + loss.item()
        
        loss_plot[e] = sum_loss
      
    return loss_plot
