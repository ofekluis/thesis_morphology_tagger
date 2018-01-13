from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
# interactive session
sess = tf.InteractiveSession()

# input and output
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# weights and biases
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# intial the variables
sess.run(tf.global_variables_initializer())

# regression model
y = tf.matmul(x,W) + b

# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# training he model, gradient decent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# the training itself
steps=1000
for i in range(steps):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# boolean array of correct results
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# get accuracy from correct_prediction by converting boolean to numeral
# and getting mean of that
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# accuracy trainset
print("accuracy train:", accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels}))

# finding out the accuracy of the testset
print("accuracy test:", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


