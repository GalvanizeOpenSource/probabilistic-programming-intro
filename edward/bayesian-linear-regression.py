#!/usr/bin/env python
"""
This example has been modified from:

http://edwardlib.org/tutorials/supervised-regression
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import edward as ed
plt.style.use('bmh')
from edward.models import Normal


def buffer_layout(ax,buff=0.01):
    """use x and y to add well spaced margins"""
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    xbuff = buff * (xmax - xmin)
    ybuff = buff * (ymax - ymin)
    ax.set_xlim(xmin-xbuff,xmax+xbuff)
    ax.set_ylim(ymin-ybuff,ymax+ybuff)


def build_toy_dataset(N, w, noise_std=0.1):
    """generate data"""
    D = len(w)
    x = np.random.randn(N, D)
    y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
    return x, y

N = 40  # number of data points
D = 10  # number of features

w_true = np.random.randn(D)
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)

X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))

qw = Normal(loc=tf.Variable(tf.random_normal([D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))


inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run(n_samples=5, n_iter=250)

y_post = ed.copy(y, {w: qw, b: qb})

# This is equivalent to
# y_post = Normal(loc=ed.dot(X, qw) + qb, scale=tf.ones(N))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))


def visualise(X_data, y_data, w, b, ax, title, n_samples=10):
  w_samples = w.sample(n_samples)[:, 0].eval()
  b_samples = b.sample(n_samples).eval()
  ax.scatter(X_data[:, 0], y_data)
  inputs = np.linspace(-8, 8, num=400)
  for ns in range(n_samples):
    output = inputs * w_samples[ns] + b_samples[ns]
    ax.plot(inputs, output)

  ax.set_title(title)
  buffer_layout(ax)
  ax.set_aspect(1./ax.get_data_ratio())
  
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Visualize samples from the prior
visualise(X_train, y_train, w, b,ax1,"Samples from prior")

# visualize samples from posterior
visualise(X_train, y_train, qw, qb, ax2, "Samples from posterior")


plt.savefig("bayes-linreg.png",dpi=400,bbox_inches = 'tight', pad_inches = 0)
plt.show()
