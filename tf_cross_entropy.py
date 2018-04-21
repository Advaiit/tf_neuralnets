import tensorflow as tf

softmax_data = [0.1, 0.8, 0.1]
one_hot_data = [0.0, 1.0, 0.0]

softmax = tf.placeholder(tf.float32)
onehot = tf.placeholder(tf.float32)

cross_entropy = tf.reduce_sum(tf.multiply(onehot, tf.log(softmax)))

cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.log(softmax), labels=onehot)

with tf.Session() as session:
    print("Cross Entropy: ", session.run(cross_entropy, feed_dict={softmax: softmax_data, onehot: one_hot_data}))
    print("Cross Entropy Loss: ", session.run(cross_entropy_loss, feed_dict={softmax: softmax_data, onehot: one_hot_data}))
