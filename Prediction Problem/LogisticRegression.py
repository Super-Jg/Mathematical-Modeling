"""
Latest updated on December 4 21:06:45 2024

@author: Sparks_Shen
"""
################################ The code is still under development ################################
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# Welcome to the LogisticRegression model (Solving the problem of binary classification prediction)


class LogisticRegression:


    def instruction(self):
        print(
            "************************************************************************************************************************************************** \n",
            "You can use 'import LogisticRegression as LR'. \n",
            "LogisticRefression: Your input should be the PATH of your file for LR_training. \n",
            "The file you input must be encoded by 'utf-8'. \n",
            "LogisticRegression has separated 'features' and 'tag' from the file you input automatically. You can view them by printing '.features' and '.tag'. \n",
            "Remember that your data['tag'] must be 0/1. \n",
            "Use 'LR.fit(iterative, learning_rate)' to start training your model. The iterations you set should be no less than 100. \n",
            "USe 'LR.visualize() to print the image of the accuracy and loss during training. \n",
            "**************************************************************************************************************************************************"
        )




    def __init__(self, path, file_type):   # 'path' is the PATH of your input file
        if (file_type == "csv"):
            self.data = pd.read_csv(path, encoding='utf-8')
        elif (file_type == "xlsx"):
            self.data = pd.read_excel(path)

        else : print("The form of your file is not supported. ")
        self.features = []
        self.tag = None
        self.weights = []
        self.bias = []


        for i in range(0, len(self.data.columns)-1):    # determine the index of features and tag
            self.features.append(self.data.columns[i])
        self.tag = self.data.columns[-1]
        self.train_x = self.data[self.features].values
        self.train_y = self.data[self.tag].values
        ones = np.ones([self.train_x.shape[0], 1])
        self.train_x = tf.constant(np.concatenate((self.train_x, ones), axis=1), dtype=tf.float32)
        self.weights = tf.Variable(np.random.randn(self.train_x.shape[1], 1), dtype=tf.float32)
        self.train_y = tf.cast(tf.reshape(self.train_y, (-1, 1)), dtype=tf.float32)




    def fit(self, iterative, learning_rate):
        self.iter = iterative
        self.lr = learning_rate
        self.loss_record = []
        self.accuracy_record = []


        w = self.weights
        for _ in range(self.iter):
            with tf.GradientTape() as tape:
                result = tf.matmul(self.train_x, w)
                pred = tf.nn.sigmoid(result)    # pred = 1 / 1+e^(-result)
                a = - self.train_y * tf.math.log(pred)
                b = - (1-self.train_y) * tf.math.log(1.-pred)
                loss = - self.train_y * tf.math.log(pred) - (1-self.train_y) * tf.math.log(1.-pred)   # Cross entropy loss function
            dloss_dw = tape.gradient(loss, w)
            w.assign_sub(self.lr * dloss_dw)     # update weights

            
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred >= 0.5, 1., 0.), self.train_y), dtype = tf.float32))

            if _ % 100 == 0:          
                self.accuracy_record.append(accuracy)
                self.loss_record.append(tf.reduce_mean(loss))
                print(f"iter{_+1}         prediction_accuracy: {accuracy}")




    def visualize(self):
        plt.plot(np.arange(self.iter/100), self.accuracy_record)
        plt.title("Accuracy Record")
        plt.show()

        plt.plot(np.arange(self.iter/100), self.loss_record)
        plt.title("Loss Record")
        plt.show()
        
# #e.g.
# from LogisticRegression import LogisticRegression as LR
# model = LR("S:\\Gits files\\Standardized dataset.csv", "csv")
# model.fit(4000, 0.0001)
# model.visualize()
