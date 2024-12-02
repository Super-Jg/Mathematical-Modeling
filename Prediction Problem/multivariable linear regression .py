import numpy as np
import tensorflow as tf
def multi_logistic(X,Y,x_dim,y_dim,train_test_split=0.8,epoch=1000,lr=0.0001):
    '''
    return w   type:tf.Variable
    this function is designed to solve "multivariable linear regression" problems

    instruction:
    to use this function,x and y must be a list or a numpy array
    exp:[ [1,2,3],
          [5,2,6],
          [2,3,5] ]
    while y must be one-hot coded
    exp: imagine a scenario where we are to classify a group of people according to gender(male,female,LGBT)
    the input y should be like[ [1,0,0],            #male
                                [0,1,0],            #female
                                [0,0,1],            #LGBT
                                [0,1,0] ]           #female
    note: make sure your data is processed, any nan or inf will be fatal
    '''
    if type(X)!=np.ndarray:
        x = np.array(X).reshape(-1,x_dim)
    else:
        x = X.reshape(-1,x_dim)
    if type(Y)!=np.ndarray:
        y = np.array(Y).reshape(-1,y_dim)
    else:
        y = Y.reshape(-1,y_dim)


    if np.isnan(x).all() or np.isnan(y).all():
        raise ValueError("np.nan found, please check your data")
    if y.shape[0]!=x.shape[0]:
        raise ValueError("x{} doesn't have the same length as y{}".format(x.shape,y.shape))

    length = x.shape[0]

    #shuffle
    np.random.seed(1)
    x = tf.constant( np.concatenate([np.random.permutation(x) , np.ones(length).reshape(length,-1)],axis=1),dtype=tf.float32)
    np.random.seed(1)
    y = tf.constant(np.random.permutation(y),dtype=tf.float32)
    #train set
    tr_x = x[:int(length*train_test_split),:]
    tr_y = y[:int(length*train_test_split),:]
    #test set
    te_x = tf.constant(x[int(length*train_test_split):,:],dtype=tf.float32)
    te_y = tf.constant(y[int(length*train_test_split):,:],dtype=tf.float32)

    arr_keep = []
    loss_keep = []
    w = tf.Variable(np.random.randn(x_dim+1,y_dim),dtype=tf.float32)
    for _ in range(epoch):
        with tf.GradientTape() as tape:
            z = tr_x @ w
            predict = tf.nn.softmax(z)
            loss = -tr_y * tf.math.log(predict) - (1 - tr_y) * tf.math.log(1 - predict)
        dl_dw = tape.gradient(loss, w)
        a = tf.argmax(predict, axis=1)
        b = tf.argmax(tr_y, axis=1)
        accr = tf.reduce_mean(tf.cast(tf.equal(a, b), dtype=tf.float32))
        arr_keep.append(accr)
        w.assign_sub(lr * dl_dw)
        loss_keep.append(tf.reduce_mean(loss).numpy())
        if _ % 10 == 0:
            print("epoch{}:    ACCR:{}".format(_, accr))
    import matplotlib.pyplot as plt
    plt.plot(np.arange(epoch), arr_keep)
    plt.title("ACCR")
    plt.show()

    #test result
    predict = tf.nn.softmax(te_x @ w)
    a = tf.argmax(predict,axis=1)
    b = tf.argmax(te_y,axis=1)
    accr = tf.reduce_mean(tf.cast(tf.equal(a, b), dtype=tf.float32))
    print("test set accuracy:",accr.numpy())
    return w

#this is an example
import pandas as pd
data = pd.read_csv(".\Iris Dataset.csv")
data['species'] = data['species'].replace({'setosa':0,
                                           'versicolor':1,
                                           'virginica':2})
for i in range(4):
    mean = np.mean(data.iloc[:,i])
    std = np.std(data.iloc[:,i])
    data.iloc[:,i]= (data.iloc[:,i]-mean)/std
x = data.iloc[:,0:4]
y = data.iloc[:,-1]
y = tf.one_hot(indices = y,depth = 3)

multi_logistic(x,y,x_dim=4,y_dim=3,epoch=5000)
