import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Prepare data on which to perform linear regression

num_points = 1000
vector_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vector_set.append([x1, y1])

x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]

#plt.plot(x_data, y_data, 'ro', label='Original Data')
#plt.legend()
#plt.show()

#Initialize weights, bias and regression formula
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

#Get mean squared error
loss = tf.reduce_mean(tf.square(y - y_data))

#Initialize gradient descent optimizer to minimize loss
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

#Run 16 epochs
for i in range(16):
    session.run(train)
    print(i, 'W: ', session.run(W), 'b: ', session.run(b))
    print(i, 'Loss: ', session.run(loss))

    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, x_data * session.run(W) + session.run(b))
    plt.xlabel('x')
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.ylabel('y')
    plt.legend()
    plt.show()

