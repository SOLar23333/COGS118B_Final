import numpy as np
import util
import math

class Activation():

    def __init__(self, activation_type = "ReLU"):
        if activation_type not in ["tanh", "ReLU","output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        self.activation_type = activation_type

        self.x = None    

    def tanh(self, x):
        self.x = x
        return np.tanh(x)

    def ReLU(self, x):
        self.x = x
        return np.maximum(x, 0)

    def output(self, x):
        self.x = x
        x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
        return x / np.sum(x, axis=1).reshape(-1, 1)

    def grad_tanh(self,x):
        return 1 - self.tanh(x) * self.tanh(x)

    def grad_ReLU(self,x):
        return np.ones(x.shape) * (x > 0)

    def grad_output(self, x):
        return 1

    def __call__(self, z):
        return self.forward(z)

    def forward(self, z):
        if self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        if self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)



class Layer():

    def __init__(self, in_units, out_units, activation):
        np.random.seed(13)

        self.w = 0.01 * np.random.random((in_units + 1, out_units))
        self.x = None    # Input to forward # N * i+1
        self.a = None    # Output without activation 
        self.z = None    # Output After Activation
        self.dw = 0      # Gradient
        self.dw_avg = 0  # Running avg gradient for momentum


        self.activation = activation   

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.a = x @ self.w
        self.z = self.activation(self.a)
        return self.z

    def backward(self, deltaCur, learning_rate, momentum_gamma=0, regularization_type=0, regularization_penalty=0, gradReqd=True):
        #deltaCur: N * j
        #self.x: N * i+1
        #self.dw: i+1 * j
        #self.a: N * j
        #newDelta = N * j @ i+1 * j

        deltaCur = np.multiply(self.activation.backward(self.a), deltaCur)
        self.dw = self.x.T @ deltaCur / self.x.shape[0]
        newDelta = deltaCur @ self.w.T

        #regularization
        if regularization_type == 2:
            self.dw -= 2 * regularization_penalty * self.w
        elif regularization_type == 1:
            self.dw -= regularization_penalty * np.sign(self.w)
        
        #momentum
        self.dw_avg = momentum_gamma * self.dw_avg + (1 - momentum_gamma) * learning_rate * self.dw
        self.w += self.dw_avg

        return newDelta[:, :-1]  #drop bias


class Neuralnetwork():
    def __init__(self, config):
        self.layers = []  
        self.num_layers = len(config['layer_specs']) - 1  
        self.x = None  
        self.y = None        
        self.targets = None  
        self.reg = config['regularization_type']
        self.reg_penalty = config['L2_penalty']
        self.learning_rate = config['learning_rate']
        self.apply_momentum = config['momentum']
        self.momentum_gamma = config['momentum_gamma']

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation'])))
            elif i == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output")))

    def __call__(self, x, targets=None):
        return self.forward(x, targets)


    def forward(self, x, targets=None):
        self.x = x
        self.targets = targets
        for layer in self.layers:
            x = layer(x)
            x = util.append_bias(x)
        
        x = x[:, :-1]
        self.y = x
        
        loss = self.loss(self.y, targets) / targets.shape[0]

        accuracy = util.calculateCorrect(self.y, self.targets)
        return loss, accuracy


    def loss(self, logits, targets):
        self.targets = targets
        sum_weight_square = 0
        sum_weight_abs = 0

        loss = -np.sum(targets * np.log(logits))
        return loss

    def backward(self, gradReqd=True):
        delta = self.targets - self.y

        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate, self.momentum_gamma, self.reg, self.reg_penalty)






