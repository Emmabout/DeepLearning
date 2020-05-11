import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
### ONLY FOR PLOTTING THE RESULTS
import matplotlib.pyplot as plt
from networks import *
from training import *
from others import *

import dlc_practical_prologue as prologue

print_errors = False

mini_batch_size = 100
N = 1000 #number of pairs
nb_epoch = 25
iterations = 10

########################################################################
########################################################################
########################################################################

#SHALLOW MODEL WITH OPTIMIZER
"""We tried several combinaisons of parameters for convolutionnal networks:
- shallow model without optimizer
- shallow model with optimizer
- shallow model with batch normalization
- deep model with optimizer
But the shallow model with optimizer was performing best (see report)."""
train_errors = []
test_errors = []
loss_plot_shallow = [0]*iterations
plt.figure("Shallow model with optimizer")
print("---------SHALLOW MODEL WITH OPTIMIZER---------")    
progressBar(0,iterations)

for i in range(iterations):
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
    train_input/=255
    test_input/=255
    new_train_target, new_test_target = reshape_targets(train_target, test_target)
    
    model = Shallow_Net(64)
    loss_plot_shallow[i] = train_model(model, train_input, new_train_target, L=0, value=0, lr=0.001, loss='MSE')
    progressBar(i+1,iterations)

    nb_train_errors = compute_nb_errors(model, train_input, new_train_target, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, new_test_target, mini_batch_size)
    
    if print_errors:
        print('train error Shallow_Net {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / test_input.size(0),
                                                  nb_train_errors, test_input.size(0)))
        print('test error Shallow_Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    train_errors.append(nb_train_errors)
    test_errors.append(nb_test_errors)

analyse_results(train_errors, test_errors)


#SIAMESE NETWORK
model = Siamese_net()
train_errors = []
test_errors = []
loss_plot_siam = [0]*iterations
print("---------SIAMESE NETWORK---------")
progressBar(0,iterations)
for i in range(iterations):
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
    train_input/=255
    test_input/=255
    new_train_target, new_test_target = reshape_targets(train_target, test_target)
    #train_model(model, train_input, new_train_target, lr,0,0)
    loss_plot_siam[i] = train_model(model, train_input, new_train_target, L=0, value=0, lr=0.001, loss='BCE')
    progressBar(i+1,iterations)
    nb_train_errors = compute_nb_errors_targets_siam(model, train_input, new_train_target)
    nb_test_errors = compute_nb_errors_targets_siam(model, test_input, new_test_target)
    
    if print_errors :
        print('train error Siamese_net {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                          nb_train_errors, train_input.size(0)))
        print('test error Siamese_net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                            nb_test_errors, test_input.size(0)))
    train_errors.append(nb_train_errors)
    test_errors.append(nb_test_errors)
analyse_results(train_errors, test_errors)

#SIAMESE + AUXILIARY LOSS
train_errors = []
test_errors = []
print("---------SIAMESE + AUXILIARY LOSS---------")
loss_plot_aux = [0]*iterations
progressBar(0,iterations)
for k in range(iterations):
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
    #normalize the input
    train_input/=255
    test_input/=255
    progressBar(k+1,iterations)
    
    new_train_classes, new_test_classes = reshape_classes(train_classes, test_classes)
    new_test_target, new_train_target = reshape_targets_aux(train_target, test_target)
    
    model = Siamese_net_auxiliary(64)
    loss_plot_aux[k] = train_model_aux(model, train_input, train_classes, train_target, 0.005,0,0)
    
    nb_train_errors = compute_nb_errors_targets_aux(model, train_input, new_train_target)
    nb_test_errors = compute_nb_errors_targets_aux(model, test_input, new_test_target)
    nb_train_errors_class = compute_nb_errors_classes(model, train_input, new_train_classes)
    nb_test_errors_class = compute_nb_errors_classes(model, test_input, new_test_classes)
    
    if print_errors:
        print('train error targets {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                          nb_train_errors, 2000))
        print('test error targets {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                        nb_test_errors, 2000))
        print('train error classes {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors_class) / train_input.size(0),
                                                          nb_train_errors_class, 2000))
        print('test error classes {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors_class) / test_input.size(0),
                                                        nb_test_errors_class, 2000))
    
    train_errors.append(nb_train_errors)
    test_errors.append(nb_test_errors)
analyse_results(train_errors, test_errors)

plt.figure("Shallow model with optimizer")
for i in range(len(loss_plot_shallow)):
    plt.plot(loss_plot_shallow[i])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolution of loss, Shallow model with optimizer")

plt.figure("Siamese network")
for i in range(len(loss_plot_siam)):
    plt.plot(loss_plot_siam[i])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolution of loss, Siamese network")

plt.figure("Siamese network and auxiliary loss")
for i in range(len(loss_plot_aux)):
    plt.plot(loss_plot_aux[i])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolution of loss, Siamese network + auxiliary loss")
plt.show()