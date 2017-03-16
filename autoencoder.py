#I have dealt with MNIST a simple image recognition dataset

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#We define generic functions for both convolution and pooling
#strides and pooling window size is given
#it can also be modified depending on examples.

def conv2D(x,w) :
	return tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME')
def maxpool2d_2x2(x) :
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
def weight_var(shape) :
	init = tf.truncated_normal(shape,stddev = 0.05);	
	return tf.Variable(init);
def bias_var(shape) :
	init = tf.constant(0.1,shape=shape);	
	return tf.Variable(init);

#placeholder variables for input and output
#input a array of 784 boolean values
#output 10 classes...represented by one-hot technique 

x = tf.placeholder(tf.float32,shape=[None,784])
y_ori = tf.placeholder(tf.float32,shape=[None,10])

#we reshape the 784*1 array to 28*28 (it helps in convolution and pooling)
#convolution layer basically extracts the features out of an image
x_in = tf.reshape(x,[-1,28,28,1])
weight_conv1 = weight_var([5,5,1,32])
bias_conv1 = bias_var([32])
l_conv1 = tf.nn.relu(conv2D(x_in,weight_conv1)+bias_conv1)
weight_conv2 = weight_var([5,5,32,64])
bias_conv2 = bias_var([64])
l_conv2 = tf.nn.relu(conv2D(l_conv1,weight_conv2)+bias_conv2)
weight_rev_conv1 = weight_var([5,5,64,32])
bias_conv3 = bias_var([32])
l_conv3 = tf.nn.relu(conv2D(l_conv2,weight_rev_conv1)+bias_conv3)
weight_rev_conv2 = weight_var([5,5,32,1])
bias_conv4 = bias_var([1])
l_conv4 = tf.nn.relu(conv2D(l_conv3,weight_rev_conv2)+bias_conv4)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = x_in,logits = l_conv4))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(x_in,1),tf.argmax(l_conv4,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())
#Basically the training begins

for i  in range(3) :
	batch = mnist.train.next_batch(50)
	if i%1 == 0:
		train_accuracy = accuracy.eval(feed_dict = { x:batch[0],y_ori:batch[1]})
		print("accuracy at step %d is %f"%(i,train_accuracy))
	train_step.run(feed_dict = { x:batch[0],y_ori:batch[1]})

print(l_conv4.eval(feed_dict={x: mnist.test.next_batch(1)[0]}))
x_i = mnist.test.next_batch(1)[0]
#print(x_i)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n = np.reshape(x_i,(28,28))
print(n)
fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.imshow(n)
fig.show()
plt.draw()
plt.waitforbuttonpress()

