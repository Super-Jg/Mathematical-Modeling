import tensorflow as tf
import numpy as np


class LSTM():
    def __init__(self,name):
        self.short_mem = 0
        self.long_mem = 0
        self.input_dim = 1
        self.output_dim = 1
        self.name=name
        # forget gate
        self.w_f_input2sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.w_f_shortmen2sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.b_f = tf.Variable(np.random.randn(), dtype=tf.float32)
        # input gate
        self.w_i_shortmen2percentage_sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.w_i_input2percentage_sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.b_i_percentage = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.w_i_shortmen2sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.w_i_input2sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.b_i = tf.Variable(np.random.randn(), dtype=tf.float32)

        # output gate
        self.w_o_shortmen2sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.w_o_input2sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.b_o = tf.Variable(np.random.randn(), dtype=tf.float32)



    def compile(self,dims):
        self.input_dim=dims[0]
        self.output_dim = dims[1]
        self.variables = [self.w_f_input2sum, self.w_f_shortmen2sum, self.b_f, self.w_i_shortmen2percentage_sum,
                          self.w_i_input2percentage_sum, self.b_i_percentage,
                          self.w_i_shortmen2sum, self.w_i_input2sum, self.b_i, self.w_o_shortmen2sum,
                          self.w_o_input2sum, self.b_o]

    def train(self,X):
        '''
        :param X: a list or numpy array
        :param input_dim: only 1 is OK
        return: an array of outputs
        '''
        ans = []
        if self.input_dim >=2:
            raise ValueError("LSTM doesn't accept x with dimension over 1")
        for x in X:
            self.long_mem = (self.long_mem * tf.nn.sigmoid(self.short_mem * self.w_f_shortmen2sum + x * self.w_f_input2sum + self.b_f)
            + tf.nn.sigmoid(self.short_mem*self.w_i_shortmen2percentage_sum+x*self.w_i_input2percentage_sum+self.b_i_percentage)*tf.nn.tanh(self.short_mem * self.w_i_shortmen2sum + x * self.w_i_input2sum + self.b_i))
            self.short_mem = (tf.nn.sigmoid(self.short_mem * self.w_o_shortmen2sum + x * self.w_o_input2sum + self.b_o)
                             * tf.nn.tanh(self.long_mem))
            ans.append(self.short_mem)
        return tf.cast(ans,dtype=tf.float32)









