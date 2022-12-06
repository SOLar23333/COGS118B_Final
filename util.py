from base64 import encode
from cProfile import label
import copy
import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd


def load_config(path):
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)



def normalize_data(inp):
    inp = np.array(inp, dtype=float)
    channels = np.reshape(inp, (-1, 3, 32*32))
    
    # do z-score normalization on each channel
    chnl0 = channels[:,0,:]
    u_array = np.mean(chnl0, axis = 1)
    sd_array = np.std(chnl0, axis = 1)
    for i in range(chnl0.shape[0]):
        chnl0[i] = (chnl0[i] - u_array[i]) / sd_array[i]
    channels[:,0,:] = chnl0
    
    chnl1 = channels[:,1,:]
    u_array = np.mean(chnl1, axis = 1)
    sd_array = np.std(chnl1, axis = 1)
    for i in range(chnl1.shape[0]):
        chnl1[i] = (chnl1[i] - u_array[i]) / sd_array[i]
    channels[:,1,:] = chnl1
    
    chnl2 = channels[:,2,:]
    u_array = np.mean(chnl2, axis = 1)
    sd_array = np.std(chnl2, axis = 1)
    for i in range(chnl2.shape[0]):
        chnl2[i] = (chnl2[i] - u_array[i]) / sd_array[i]
    channels[:,2,:] = chnl2
    
    normalized_data = np.reshape(channels, (inp.shape))
    return normalized_data


def one_hot_encoding(labels, num_classes=10):
    arr = np.eye(len(labels), num_classes, dtype=int)[labels]
    newArr = np.reshape(arr, (labels.shape[0], num_classes))
    return newArr

def one_hot_decoding(y):
    #each row is a one_hot_encoding
    return np.argmax(y, axis=1)


def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def calculateCorrect(y,t):
    predictions = np.argmax(y, axis=1)
    targets = one_hot_decoding(t);
    return np.sum(predictions == targets)  / predictions.shape[0]



def append_bias(X):
    newColumn = np.ones((X.shape[0], 1), dtype=float)
    X = np.concatenate((X, newColumn), axis=1)
    return X




def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop):

    saveLocation="./plots/"

    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'b', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='b',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(saveLocation+"loss.eps")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'b', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='b', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(saveLocation+"accuarcy.eps")
    plt.show()

    #Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(saveLocation+"trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(saveLocation+"valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(saveLocation+"trainEpochAccuracy.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(saveLocation+"valEpochAccuracy.csv")



def createTrainValSplit(x_train,y_train):

    
    # shuffle the data
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]
    
    # split the set into train and val
    train_num = int(x_train.shape[0] * 0.8)
    train_images = x_train[:train_num, :]
    train_labels = y_train[:train_num]
    val_images = x_train[train_num:, :]
    val_labels = y_train[train_num:]
    return train_images, train_labels, val_images, val_labels



def shuffle(x_train, y_train):
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    return x_train[idx], y_train[idx]



def load_data(path):
    """
    Loads, splits our dataset- CIFAR-10 into train, val and test sets and normalizes them

    args:
        path: Path to cifar-10 dataset
    returns:
        train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, "cifar-10-batches-py")

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    for i in range(1,6):
        images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
        data = images_dict[b'data']
        label = images_dict[b'labels']
        train_labels.extend(label)
        train_images.extend(data)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels).reshape((len(train_labels),-1))
    train_images, train_labels, val_images, val_labels = createTrainValSplit(train_images,train_labels)

    train_normalized_images =  normalize_data(train_images)
    train_one_hot_labels = one_hot_encoding(train_labels)

    val_normalized_images = normalize_data(val_images)
    val_one_hot_labels = one_hot_encoding(val_labels)

    test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
    test_data = test_images_dict[b'data']
    test_labels = test_images_dict[b'labels']
    test_images = np.array(test_data)
    test_labels = np.array(test_labels).reshape((len(test_labels),-1))
    test_normalized_images= normalize_data(test_images)
    test_one_hot_labels = one_hot_encoding(test_labels)
    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels
