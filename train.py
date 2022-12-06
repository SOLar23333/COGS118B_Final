import copy
import util
from neuralnet import *


def train(model, x_train, y_train, x_valid, y_valid, config):

    # Read in the esssential configs
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    early_stop = config["early_stop"]
    early_stop_epoch = config["early_stop_epoch"]
    
    # lists for plotting
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    
    early_stop_position = epochs-1
    stopping_count = 0
    best_loss = 10

    for i in range(epochs):
        print("---- training epoch:", i+1, "----")
        train_batch_count = 0
        total_train_loss = 0
        total_train_accuracy = 0

        # shuffle the data
        x_train, y_train = util.shuffle(x_train, y_train)
        
        # stochastic gradient descent
        for (mini_train_data, mini_train_labels) in util.generate_minibatches((x_train, y_train), batch_size):
            train_loss, train_accuracy = model.forward(mini_train_data, mini_train_labels)
            model.backward()
            # calculate acc and loss
            train_batch_count += 1
            total_train_loss += train_loss
            total_train_accuracy += train_accuracy
        
        epoch_train_loss = total_train_loss / train_batch_count
        epoch_train_accuracy = total_train_accuracy / train_batch_count
        
        epoch_valid_loss, epoch_valid_accuracy = model.forward(x_valid, y_valid)
        
        # add data to the plot list
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_accuracy)
        valid_loss_list.append(epoch_valid_loss)
        valid_acc_list.append(epoch_valid_accuracy)
        
        print("loss:", epoch_valid_loss)
        print("accuracy:", epoch_valid_accuracy)
        
        # early stopping condition check
        if early_stop:
            if epoch_valid_loss > best_loss:
                stopping_count += 1
                print("stopping count:", stopping_count)
                if stopping_count >= early_stop_epoch:
                    early_stop_position = i
                    best_model = copy.deepcopy(model)
                    early_stop = False
            else:
                stopping_count = 0
                best_loss = epoch_valid_loss
        
        if i == epochs - 1:
            util.plots(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, early_stop_position)
            if early_stop == True:
                best_model = copy.deepcopy(model)
    return best_model

def modelTest(model, X_test, y_test):
    return model.forward(X_test, y_test)


