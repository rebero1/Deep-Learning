# USING TENSORFLOW FOR AUTOENCODERING
import numpy.random as rnd
import numpy as np

rnd.seed(42)
m = 200
w1,w2=.1,.3
noise = .1



angles = rnd.rand(m)*3*np.pi/2-.5


data=np.empty((m,3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

#%%
from sklearn.preprocessing import StandardScaler


model = StandardScaler()
X_train = model.fit_transform(data[:100])
X_test = model.fit_transform(data[100:])
#%%

import tensorflow as tf   
from tensorflow.layers import dense
tf.reset_default_graph()
n_inputs = 3
n_hidden2 = 2



n_outputs=n_inputs

learning_rate = .01


X =tf.placeholder(tf.float32,shape=(None,n_inputs))

hidden=  dense(X,n_hidden2)
n_outputs= dense(hidden,n_outputs)

loss = tf.losses.mean_squared_error(X,n_outputs)
training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init=tf.global_variables_initializer()
#%%


n_iterations=100

codings=hidden


with tf.Session() as sess:
  sess.run(init)

  for _ in range(n_iterations):
    sess.run(training_op,feed_dict={X:X_train})
  coding_vals = codings.eval(feed_dict={X:X_test})
  print(sess.run(codings))


#%%

import matplotlib.pyplot as plt
plt.plot(coding_vals[:,0],coding_vals[:,1],'b.')
plt.xlabel("$z_1$",fontsize=18)
plt.xlabel("$z_2$", fontsize=18)
plt.plot(X_test[:, 0], X_test[:, 1], 'r.')
plt.show()
