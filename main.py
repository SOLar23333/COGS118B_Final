from numpy import append
from train import *
from util import append_bias
import argparse

def main(args):

    config = util.load_config("./configs/config.yaml")

    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(path="./data/")
    x_train = append_bias(x_train)
    x_valid = append_bias(x_valid)
    x_test = append_bias(x_test)

    model = Neuralnetwork(config)
    model = train(model, x_train, y_train, x_valid, y_valid, config)

    test_loss, test_acc = modelTest(model, x_test, y_test)
    print('Test Accuracy:', test_acc, ' Test Loss:', test_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)