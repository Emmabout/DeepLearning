import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

mini_batch_size = 100
N = 1000 #number of pairs
nb_epoch = 25
iterations = 1

##############################OTHER#################################
##reshape the train_ and test_target to obtain size [1000,2]
def reshape_targets(train_target, test_target):
    new_train_target = torch.empty(1000,2)
    new_test_target = torch.empty(1000,2)
    for i in range(1000):
        if train_target[i] == 1 :
            new_train_target[i,0] = 0
            new_train_target[i,1] = 1

        else:
            new_train_target[i,0] = 1
            new_train_target[i,1] = 0

        if test_target[i] == 1:
            new_test_target[i,0] = 0
            new_test_target[i,1] = 1

        else:
            new_test_target[i,0] = 1
            new_test_target[i,1] = 0
    return new_train_target, new_test_target

###ANALYSE THE RESULTS
def analyse_results(train_errors, test_errors):
    train_errors = torch.Tensor(train_errors)
    test_errors = torch.Tensor(test_errors)
    print('\n Average train error {:0.2f}% {:0.2f}/{:d}'.format((100 * train_errors.mean()) / N,
                                                          train_errors.mean(), N))
    print("Train error standard deviation : {:0.2f}%".format((100 * train_errors.std()) / N,
                                                          train_errors.std(), N))

    print('Average test error {:0.2f}% {:0.2f}/{:d}'.format((100 * test_errors.mean()) / N,
                                                          test_errors.mean(), N))
    print("Test error standard deviation : {:0.2f}%".format((100 * test_errors.std()) / N,
                                                          test_errors.std(), N))

    train_err = [x*100 / N for x in train_errors]
    test_err = [x*100 / N for x in test_errors]

def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

def compute_nb_errors(model, input, target, mini_batch_size): 
    #target[1000], predicted_classes[100], output[100*2]
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors

def compute_nb_errors_targets_aux(model, input, target):
    nb_errors = 0
    _, output = model(input)
    _, predicted_target = output.max(1) #max probabilities of target
    
    for b in range(1000):
        if target[b,int(predicted_target[b])] <= 0:
            nb_errors = nb_errors + 1
            
    return nb_errors

def compute_nb_errors_classes(model, input, target):
    nb_errors = 0

    output,_ = model(input)
    _, predicted_classes = output.max(2)

    for b in range(input.shape[0]):
        if target[b][0][predicted_classes[b][0]] <= 0:
            nb_errors = nb_errors + 1
        if target[b][1][predicted_classes[b][1]] <= 0:
            nb_errors = nb_errors + 1

    return nb_errors

def compute_nb_errors_targets_siam(model, input, target):
    nb_errors = 0
    output = model(input)
    _, predicted_target = output.max(1) #max probabilities of target
    
    for b in range(1000):
        if target[b,int(predicted_target[b])] <= 0:
            nb_errors = nb_errors + 1
            
    return nb_errors

def reshape_targets_aux(train_target, test_target):
    new_train_target = torch.empty(1000,2)
    new_test_target = torch.empty(1000,2)
    for i in range(1000):
        if train_target[i] == 1 :
            new_train_target[i,0] = 0
            new_train_target[i,1] = 1

        else:
            new_train_target[i,0] = 1
            new_train_target[i,1] = 0

        if test_target[i] == 1:
            new_test_target[i,0] = 0
            new_test_target[i,1] = 1

        else:
            new_test_target[i,0] = 1
            new_test_target[i,1] = 0
            
    return new_test_target, new_train_target

#train_classes[1000, 2]
def reshape_classes(train_classes, test_classes):
    new_train_classes = torch.zeros(1000, 2, 10)
    new_test_classes = torch.zeros(1000, 2, 10)

    for i in range(train_classes.shape[0]): #
        new_train_classes[i][0][train_classes[i][0]] = 1
        new_train_classes[i][1][train_classes[i][1]] = 1

    for i in range(test_classes.shape[0]):
        new_test_classes[i][0][test_classes[i][0]] = 1
        new_test_classes[i][1][test_classes[i][1]] = 1

    return new_train_classes, new_test_classes
