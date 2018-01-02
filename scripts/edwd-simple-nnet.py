#!/usr/env/python

"""
simple neural network example to model cosine data
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from edward.models import Normal
import edward as ed
plt.style.use('bmh')

def buffer_layout(ax,buff=0.01):
    """use x and y to add well spaced margins"""
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    xbuff = buff * (xmax - xmin)
    ybuff = buff * (ymax - ymin)
    ax.set_xlim(xmin-xbuff,xmax+xbuff)
    ax.set_ylim(ymin-ybuff,ymax+ybuff)

## create the data
print("\t ...specify data")
x_train = np.linspace(-3, 3, num=50)
y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=50)
x_train = x_train.astype(np.float32).reshape((50, 1))
y_train = y_train.astype(np.float32).reshape((50, 1))

## plot data
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

ax.plot(x_train,y_train,color='darkblue',markersize=10,linestyle='none',marker='s')
buffer_layout(ax)
ax.set_aspect(1./ax.get_data_ratio())

## specify the weights and biases of a simple nerural networks
print("\t ...specify base model")
W_0 = Normal(loc=tf.zeros([1, 2]), scale=tf.ones([1, 2]))
W_1 = Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]))
b_0 = Normal(loc=tf.zeros(2), scale=tf.ones(2))
b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

## use tanh nonlinearities
x = x_train
y = Normal(loc=tf.matmul(tf.tanh(tf.matmul(x, W_0) + b_0), W_1) + b_1, scale=0.1)

## Specify a normal approximation over the weights and biases (for variational inference)
print("\t ...specify priors")
qW_0 = Normal(loc=tf.Variable(tf.zeros([1, 2])),
              scale=tf.nn.softplus(tf.Variable(tf.zeros([1, 2]))))
qW_1 = Normal(loc=tf.Variable(tf.zeros([2, 1])),
              scale=tf.nn.softplus(tf.Variable(tf.zeros([2, 1]))))
qb_0 = Normal(loc=tf.Variable(tf.zeros(2)),
              scale=tf.nn.softplus(tf.Variable(tf.zeros(2))))
qb_1 = Normal(loc=tf.Variable(tf.zeros(1)),
              scale=tf.nn.softplus(tf.Variable(tf.zeros(1))))

## carry out variaitonal inference
print("\t ...performing inference")
inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1}, data={y: y_train})
inference.run(n_iter=1000)

plt.show()

print("done")
