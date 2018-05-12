import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

num_points = 2000
clusters = []

#Create two clusters with centroids at 3.0 and 0.0, with probability distribution 0.5
for i in range(num_points):
    if np.random.random() > 0.5:
        clusters.append([np.random.normal(3.0, 0.5), np.random.normal(3.0, 0.5)])
    else:
        clusters.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])

#dataFrame = pd.DataFrame({"x":[v[0] for v in clusters], "y": [v[1] for v in clusters]})

# sns.lmplot(x="x", y="y", data=dataFrame, fit_reg=False, size=9)
# plt.show()

vectors = tf.constant(clusters)
k = 4
centroids = tf.Variable(tf.slice(tf.random_shuffle(clusters), [0, 0], [k, -1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

print ('Exapanded vectors: ', expanded_vectors.get_shape())
print ('Expanded centroids: ', expanded_centroids.get_shape())

#Euclidean distance
#d^2 = (x2 - x1)^2 + (y2 - y1)^2
diff = tf.subtract(expanded_vectors, expanded_centroids)
print ('Difference shape: ', diff.get_shape())
sqr = tf.square(diff)
print ('Square shape: ', sqr.get_shape())
distances = tf.reduce_sum(sqr, 2)
print ('Distances shape: ', distances.get_shape())
assignments = tf.argmin(distances, 0)
print ('Assignments shape: ', assignments.get_shape())

#Calculate distance in single line
#assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2), 0)

#Calculate new centroids
means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in range(k)], 0)

updated_centroids = tf.assign(centroids, means)

init = tf.global_variables_initializer()

session = tf.Session()

session.run(init)

for step in range(100):
    _, centroid_values, assignment_values = session.run([updated_centroids, centroids, assignments])
    data = {"x": [], "y": [], "cluster": []}

print ('Assignment values shape: ', np.array(assignment_values).shape)
print ('Assignment values: ', assignment_values)

for i in range(len(assignment_values)):
    data["x"].append(clusters[i][0])
    data["y"].append(clusters[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()

print (centroid_values)