import tensorflow as tf

x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

a = tf.placeholder("float", [None, 3])
b = a * 2

input_array = [[1, 2, 3], [4, 9, 9],]

#returns a tensor with random values from normal distribution, usually used to initialize weights
normal_weights = tf.truncated_normal((2, 3))

#Softmax graph
logit_data = [2.0, 1.0, 1.0]
logits = tf.placeholder(tf.float32)
softmax = tf.nn.softmax(logits)

with tf.Session() as session:
    print(session.run(x, feed_dict={x: 'Hello World'}))
    print(session.run(y, feed_dict={x: 'Hello World', y: 10, z: 10.5}))
    print(session.run(b, feed_dict={a: input_array}))
    print(session.run(normal_weights))
    print('Softmax: ', session.run(softmax, feed_dict={logits: logit_data}))
