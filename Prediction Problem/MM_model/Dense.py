"""
Created on Nov 27 18:38:08 2024

@author: Sparks_Shen
"""

import numpy as np
import tensorflow as tf

class Dense:

    def __init__(self,name,activation='sigmoid'):
        self.input_dim = None
        self.output_dim = None
        self.activation = activation
        self.name=name
    def compile(self,dims):
        self.input_dim=dims[0]
        self.output_dim = dims[1]
        self.variables = [tf.Variable(np.random.randn(self.input_dim, self.output_dim), dtype=tf.float32),
                          tf.Variable(np.random.randn(), dtype=tf.float32)]

    def train(self, input):
        # initialize weights and bias

        if self.activation == None:
            output = input @ self.variables[0] + self.variables[1]

        elif self.activation == "sigmoid":
            output = input @ self.variables[0] + self.variables[1]

        elif self.activation == "tanh":
            output = input @ self.variables[0] + self.variables[1]

        elif self.activation == "relu":
            output = input @ self.variables[0] + self.variables[1]
        else:
            output = None
        return output


#eg:
# input = np.array([[1, 2, 3],
#                   [4, 5, 6]])
# dense = Dense([3, 4, 5, 1])
# output = dense.train(input, type='sigmoid')
# print(output)