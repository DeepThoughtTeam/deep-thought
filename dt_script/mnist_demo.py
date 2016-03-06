'''
	run_mlp takes size of hidden layer 
'''

import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_hs, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_hs[0])) # this is a basic mlp, think 2 stacked logistic regressions
    for w_h in w_hs[1:]:
    	h = tf.nn.sigmoid(tf.matmul(h, w_h))
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for u

def run_mlp(input_weight = 784, num_output = 10, hidden_weight = [12], lr = 0.001):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

	X = tf.placeholder("float", [None, input_weight])
	Y = tf.placeholder("float", [None, num_output])

	w_hs = []
	w_hs.append(init_weights([input_weight, hidden_weight[0]]))
	for i in xrange(len(hidden_weight)-1):
		w_hs.append(init_weights([hidden_weight[i], hidden_weight[i+1]]))

	w_o = init_weights([hidden_weight[-1], num_output])

	py_x = model(X, w_hs, w_o)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
	 # construct an optimizer, choice of learning rate
	train_op = tf.train.RMSPropOptimizer(lr, 0.9).minimize(cost)
	predict_op = tf.argmax(py_x, 1)

	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)

	for i in range(num_output):
	    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
	        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
	    print i, np.mean(np.argmax(teY, axis=1) ==
	                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))

def main():
	hidden_weight = [200, 35, 23]
	run_mlp(784, 10, hidden_weight, 0.005)

if __name__ == "__main__":
	main()
