import tensorflow as tf

hello_constant = tf.constant(str('Hello World!'), name='hello_constant')
constant_x = tf.constant(5, name='constant_x')
variable_y = tf.Variable(constant_x + 5, name='variable_y')

x = tf.add(10, variable_y, name='x')
y = tf.subtract(20, 1, name='y')
z = tf.multiply(10, 10, name='z')

array1 = [1, 2, 3]
array2 = [4, 5, 6]

a = tf.multiply(array1, array2)

b = tf.reduce_sum(tf.multiply(array1, array2))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print('hello constant: ',session.run(hello_constant), '\n')
    print('x: ',session.run(x), '\n')
    print('y: ', session.run(y), '\n')
    print('z: ',session.run(z), '\n')
    print('Array Multiplication: ', session.run(a), '\n')
    print('Array Reduce Sum: ', session.run(b), '\n')