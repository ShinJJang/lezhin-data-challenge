
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import tensorflow as tf
import numpy as np
import pandas as pd
import time

from numpy import genfromtxt
from scipy import stats


# In[2]:

start_time = time.time()


# In[3]:

def read_data(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)
    return df


# In[4]:

df = read_data('lezhin_public_dataset_training.tsv')


# In[5]:

# df.iloc[:, :20]
del df[7], df[8], df[16], df[18]


# In[6]:

df.describe()


# In[7]:

features = df.iloc[:, 1:].values
labels = df.iloc[:, :1].values
print(stats.describe(features).variance)
print(features.shape, labels.shape)


# In[8]:

rnd_indices = np.random.rand(len(features)) < 0.70

train_x = features[rnd_indices]
train_y = labels[rnd_indices]
test_x = features[~rnd_indices]
test_y = labels[~rnd_indices]
print("train row count : %d, test row count : %d" % (train_x.shape[0], test_x.shape[0]))

feature_count = train_x.shape[1]
label_count = train_y.shape[1]
print(feature_count, label_count)


# In[9]:

training_epochs = 90
learning_rate = 0.01
cost_history = np.empty(shape=[1],dtype=float)
nb_classes = 2

X = tf.placeholder(tf.float32,[None,feature_count])
Y = tf.placeholder(tf.int32,[None,label_count])
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)


# In[10]:

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1)), tf.Variable(tf.random_normal([shape[1]]))

def make_hidden_layer(previous_h, weight, bias, p_keep_hidden, is_dropout=True):
    h = tf.nn.relu(tf.matmul(previous_h, weight) + bias)
    if is_dropout:
        h = tf.nn.dropout(h, p_keep_hidden)
    return h

def model(X, p_keep_hidden):
    s_1 = feature_count + 2
    s_2 = feature_count + 2
    s_3 = feature_count
    
    w_h, b = init_weights([feature_count, s_1])
    w_h2, b2 = init_weights([s_1, s_2])
    w_h3, b3 = init_weights([s_2, s_3])
    w_o, b_o = init_weights([s_3, nb_classes])
    
    h = make_hidden_layer(X, w_h, b, p_keep_hidden)
    h2 = make_hidden_layer(h, w_h2, b2, p_keep_hidden)
    h3 = make_hidden_layer(h2, w_h3, b3, p_keep_hidden, False)
    
    return tf.matmul(h3, w_o) + b_o


# In[11]:

p_keep_hidden = tf.placeholder("float")

h0 = model(X, p_keep_hidden)


# In[12]:

# Cross entropy cost/loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h0, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[13]:

prediction = tf.argmax(h0, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[14]:

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
print(X.shape, Y.shape)
training_dropout_h = 0.95

batch_size = 2000
batch_length = int(train_x.shape[0] / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        for batch_num in range(batch_length):
            start_idx = batch_num * batch_size
            end_idx = (train_x.shape[0] - 1) if batch_num == batch_length - 1 else (batch_num + 1) * batch_size
                
            if batch_num % 200 == 0 or batch_num == batch_length - 1:
                print("batch num : %d / %d, index: %d ~ %d" % (batch_num, batch_length - 1, start_idx, end_idx))
            
            sess.run(optimizer, feed_dict={X: train_x[start_idx:end_idx], Y: train_y[start_idx:end_idx], p_keep_hidden: training_dropout_h})

        loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: train_x, Y: train_y, p_keep_hidden: training_dropout_h})
        cost_history = np.append(cost_history, acc)
        if step % 4 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
            
    # Test model and check accuracy
    pre = tf.argmax(h0, 1)
    test_yy = np.transpose(test_y.ravel())
    print(test_yy.shape)
    correct_prediction = tf.equal(pre, test_yy)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Test Accuracy:', sess.run(accuracy, feed_dict={X: test_x, p_keep_hidden: 1.0}))


# In[15]:

print(cost_history.shape)
plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,1])
plt.show()


# In[16]:

sess.close()
end_time = time.time()
print("processing time : %d seconds" % (end_time - start_time,))

