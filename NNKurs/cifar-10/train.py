import tensorflow as tf
import numpy as np

from sys import exit
import tensorpack.dataflow.dataset as dataset
train, test = dataset.Cifar10('train'), dataset.Cifar10('test')

# useful to reduce this number to 1000 for debugging purposes
n = 50000
x_train = np.array([train.data[i][0] for i in range(n)], dtype=np.float32)
y_train = np.array([train.data[i][1] for i in range(n)], dtype=np.int32)
x_test = np.array([ex[0] for ex in test.data], dtype=np.float32)
y_test = np.array([ex[1] for ex in test.data], dtype=np.int32)

# del(train, test)  # frees approximately 180 MB
# make one_hot 10 long array's from y
y_train = [[1.0 if y_train[j]==i else 0.0 for i in range(10)] for j in range(len(y_train))]
y_test = [[1.0 if y_test[j]==i else 0.0 for i in range(10)] for j in range(len(y_test))]

# standardization
x_train_pixel_mean = x_train.mean(axis=0)  # per-pixel mean
x_train_pixel_std = x_train.std(axis=0)   # per-pixel std
x_train -= x_train_pixel_mean
x_train /= x_train_pixel_std
x_test -= x_train_pixel_mean
x_test /= x_train_pixel_std


x = tf.placeholder(tf.float32, shape=[None, 32, 32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
		strides=[1, 2, 2, 1], padding='SAME')

def global_average_pool(x):
    return tf.reduce_mean(x,[1,2])
    #return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5, 5, 3, 10])
b_conv1 = bias_variable([10])

# reshape input pictures to 4d so we can convolve over it with our filter
x_image = tf.reshape(x, [-1, 32, 32, 3])

# ReLU + max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) # after this the image is 14x14 big.

# second layer 20 dimension for the output feature map
W_conv2 = weight_variable([4, 4, 10, 20])
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 3rd conv layer
W_conv3 = weight_variable([3, 3, 20, 30])
b_conv3 = bias_variable([30])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = global_average_pool(h_conv3)
# reshape tensor into a batch of vectors, muliply by weights add bias and apply ReLU
W_fc1 = weight_variable([30, 10])
b_fc1 = bias_variable([10])

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_pool3, keep_prob)

y_conv = tf.matmul(h_fc1_drop, W_fc1) + b_fc1


#print(y_conv)
#exit("finished")
# apply dropout to avoid overfitting

# readout layer, do softmax to get some output y, remember we have 10 number
# therefore we use 10 output dimension, for the classification.
#W_fc2 = weight_variable([1024, 10])
#b_fc2 = bias_variable([10])

#y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train in batches, using adam optimizier (instead of steepest gradient decent).
# log every 100th and use non interactive mode.

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
bSize=50
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(0,50000-bSize,bSize):
    batch = [x_train[i:i+bSize], y_train[i:i+bSize]]
    if i % bSize == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i/50, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: x_test, y_: y_test, keep_prob: 1.0}))


