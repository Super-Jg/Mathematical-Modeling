import numpy as np
import tensorflow as tf
import LSTM
import Dense
class model():
    def __init__(self,epoch,lr):
        self.epoch = epoch
        self.lr = lr

        self.variables = []

        self.layers = []
        self.x = None
        self.y = None

        self.tr_x = None
        self.te_x = None
        self.tr_y = None
        self.te_y = None

        self.interlayer_dims = []

    def compile(self,input_dim):
        '''
        temporarily only input_dim is available
        in further updates,params such as test_function(MSE,CE) will be available
        '''
        self.temp_input_dim = input_dim

    def add(self,layer,output_dim):
        layer.compile((self.temp_input_dim, output_dim))
        self.temp_input_dim = output_dim
        self.layers.append(layer)
        self.variables += layer.variables
    def data_test(self,X,Y):
        if type(X) != np.ndarray:
            self.x = np.array(X)
        else:
            self.x = X
        if type(Y) != np.ndarray:
            self.y = np.array(Y)
        else:
            self.y = Y
        if np.isnan(self.x).all() or np.isnan(self.y).all():
            raise ValueError("np.nan found, please check your data")
        if self.y.shape[0] != self.x.shape[0]:
            raise ValueError("x{} doesn't have the same length as y{}".format(self.x.shape, self.y.shape))


    def fit(self,X,Y,shuffle=False,validation=False,vali_x=None,vali_y=None):

        self.data_test(X,Y)

        if shuffle:
            seed = np.random.randint(10000)
            np.random.seed(seed)
            self.x = tf.constant(np.random.permutation(self.x),dtype=tf.float32)
            np.random.seed(seed)
            self.y = tf.constant(np.random.permutation(self.y), dtype=tf.float32)

        if validation:
            self.te_x = vali_x
            self.te_y = vali_y
        self.tr_x = self.x
        self.tr_y = self.y

    def summary(self):
        print("current variable num:",len(self.variables))
        print("layers:",[i.name for i in self.layers])


    def train(self):
        for _ in range(self.epoch):
            with tf.GradientTape() as tape:

                temp = tf.constant(self.tr_x,dtype=tf.float32)
                for layer in self.layers:
                    temp = layer.train(temp)
                loss = tf.reduce_mean(tf.math.square(temp-self.tr_y))

            dl_dw = tape.gradient(loss,self.variables)
            for d in range(len(dl_dw)):

                self.variables[d].assign_sub(dl_dw[d]*self.lr)
            if _%20==0:
                print(print("epoch:{}    Loss:{}".format(_, loss)))
        print("Training success")
        return 1

    def predict(self,input):
        temp = tf.constant(input, dtype=tf.float32)
        for layer in self.layers:
            temp = layer.train(temp)
        return temp

