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
x_out1 = maxpool2d_2x2(l_conv1)
weight_conv2 = weight_var([5,5,32,64])
bias_conv2 = bias_var([64])
l_conv2 = tf.nn.relu(conv2D(x_out1,weight_conv2)+bias_conv2)
x_out2 = maxpool2d_2x2(l_conv2)
#So basically 2 layers of conv-pool ...(for this example)
#now we feed the feature set to a fully connected neural network
weight_ful1 = weight_var([7*7*64,1024])
bias_ful1 = bias_var([1024])
x_from_conv = tf.reshape(x_out2,[-1,7*7*64])
x_from_ful1 = tf.nn.relu(tf.matmul(x_from_conv,weight_ful1)+bias_ful1)
weight_ful2 = weight_var([1024,10])
bias_ful2 = bias_var([10])
y = tf.matmul(x_from_ful1,weight_ful2)+bias_ful2
#We calculate the cross entropy
#and apply the optimization function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_ori,logits = y))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_ori,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())
#Basically the training begins
for i  in range(2000) :
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict = { x:batch[0],y_ori:batch[1]})
		print("accuracy at step %d is %f"%(i,train_accuracy))
	train_step.run(feed_dict = { x:batch[0],y_ori:batch[1]})
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

