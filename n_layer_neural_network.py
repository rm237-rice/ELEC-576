__author__ = 'rhia_malhotra'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

from three_layer_neural_network import NeuralNetwork, generate_data, plot_decision_boundary

########################################################################################################################
########################################################################################################################
# YOUR ASSIGNMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN AN N-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
def softmax(x):
    '''
    softmax computes the softmax of the input x
    :param x: input data
    :return: softmax of the input x
    '''
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class Layer(NeuralNetwork):
    """
    This class builds and trains a layer of neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, nn_layer_idx, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param nn_layer_idx: index of the layer
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        super().__init__(nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type, reg_lambda, seed)
        self.nn_layer_idx = nn_layer_idx
        self.W = np.random.randn(nn_input_dim, nn_hidden_dim) / np.sqrt(nn_input_dim)
        self.b = np.zeros((1, nn_hidden_dim))

    def feedforward(self, X, actFun):
        '''
        computes the output of the layer given input data X and activation function actFun
        :param X: input data
        :param actFun: activation function
        :return:
        '''
        self.z = X @ self.W + self.b
        self.a = actFun(self.z)
        return None

    def backprop(self, output_error, num_examples, prev_layer):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param output_error: error in the output layer
        :param num_examples: number of examples in the batch
        :param prev_layer: previous layer object
        :return: gradients dL/dW, dL/db
        '''
        # IMPLEMENT YOUR BACKPROP HERE

        a1 = prev_layer.a
        z1 = prev_layer.z

        dW2 = a1.T @ output_error / num_examples
        db2 = np.sum(output_error, axis=0, keepdims=True) / num_examples # dz_db = 1
        input_error = output_error @ self.W.T * self.diff_actFun(z1, type=self.actFun_type)
    
        return dW2, db2, input_error

class DeeperNeuralNetwork(NeuralNetwork):
    """
    This class builds and trains a deeper neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, nn_layers, actFun_type='tanh', reg_lambda=0.01, seed=0):            
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units (layer size)
        :param nn_output_dim: output dimension
        :param nn_layers: the number of layers
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''        
        super().__init__(nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type, reg_lambda, seed)
        self.nn_layers = nn_layers

        self.layers = []
        for l in range(nn_layers):
            if l == 0:
                self.layers.append(Layer(nn_input_dim, nn_hidden_dim, nn_output_dim, l, actFun_type, reg_lambda, seed))
            else:
                self.layers.append(Layer(nn_hidden_dim, nn_hidden_dim, nn_output_dim, l, actFun_type, reg_lambda, seed))
        
    def feedforward(self, X, actFun):
        '''
        feedforward builds an N-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''
        for l in range(self.nn_layers):
            if l == 0:
                self.layers[l].feedforward(X, actFun)
                # print(self.layers[0].a)
            elif l == self.nn_layers-1:
                self.layers[l].feedforward(self.layers[l-1].a, softmax)
            else:
                self.layers[l].feedforward(self.layers[l-1].a, actFun)

        self.probs = self.layers[-1].a

        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss
        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        # print(self.probs[range(num_examples), y])
        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))

        # Add l2 regularization term to loss (optional))
        data_loss += self.reg_lambda / 2 * sum(np.sum(np.square(layer.W)) for layer in self.layers)

        return (1. / num_examples) * data_loss
    
    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: array of gradients dL/dW for each layer, array of gradients dL/db for each layer
        '''
        dW_layers = []
        db_layers = []
        
        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        y_probs = np.zeros_like(self.probs)
        y_probs[range(num_examples), y] = 1

        output_error = self.probs - y_probs # softmax derivative for last layer

        for l in np.arange(self.nn_layers-1, 0, -1):
            dW, db, output_error = self.layers[l].backprop(output_error, num_examples, self.layers[l-1]) # input_error for this layer is the output_error for the next layer
            dW_layers.insert(0, dW) # prepend to front of list since we are counting backwards
            db_layers.insert(0, db) 

        return dW_layers, db_layers

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW_layers, db_layers = self.backprop(X, y)

            # Add derivatives of regularization terms (b doesn't have regularization terms)
            for l in range(1, self.nn_layers-1):
                # print("dW shape: ", dW_layers[l].shape)
                # print("W shape: ", self.layers[l].W.shape)
                dW_layers[l] += self.reg_lambda * self.layers[l].W
            
            # Gradient descent parameter update
            for l in range(1, self.nn_layers-1):
                self.layers[l].W += -epsilon * dW_layers[l]
                self.layers[l].b += -epsilon * db_layers[l]
            
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    model = DeeperNeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, nn_layers=3, actFun_type='tanh')
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()