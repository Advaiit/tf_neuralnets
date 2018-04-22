#Classify on MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

#Pixels of an image in MNIST dataset, 28 x 28
feature_count = 784

#Classify in 10 digits
labels_count = 10
batch_size = 128
epochs = 10
learning_rate = 0.5

features = tf.placeholder(tf.float32, [None, feature_count])
labels = tf.placeholder(tf.float32, [None, labels_count])
weights = tf.Variable(tf.truncated_normal([feature_count, labels_count]))
biases = tf.Variable(tf.zeros(labels_count))

logits = tf.add(tf.matmul(features, weights), biases)
prediction = tf.nn.softmax(logits)

cross_entropy = tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

#Calculate training loss
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(init)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batchx, batchy = mnist.train.next_batch(batch_size=batch_size)
            _, c = session.run([optimizer, loss], feed_dict={features: batchx, labels: batchy})
            print("c: ", c, "\n")
            avg_cost += c / total_batch
        print("Epoch: ", epoch + 1, " cost: ", avg_cost)
    print(session.run(accuracy, feed_dict={features: mnist.test.images, labels:mnist.test.labels}))




