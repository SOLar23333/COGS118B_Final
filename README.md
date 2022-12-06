# CSE 118B Final: Multi-layer Neural Networks that Classify CIFAR-10 Dataset

**Jared Zhang A15889667**

## Introduction
This project contains a multi-layer nueral network that is able to classify CIFAR-10 dataset with about 47.2% accuracy. The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, which are airplanes, automobile, birds, cat, deer, dog, frog, horse, ship, and truck. We split the data into two sets, training set and testing set. Then the network is trained on the training dataset with the specified parameters in config file. Two figures (loss over time and accuracy over time) will be plotted when the training is done. After that, we use the best model out of the training to test on the testing dataset and output loss and accuracy.

No high-level machine learning / deep learning package was used.


## Usage
**0. Due to potential low speed of downloading the dataset, the dataset has been pre-stored.**

**1. Specify the configuration of the network in config.yaml**

**2. Run main.py**  
`$ python main.py`  
