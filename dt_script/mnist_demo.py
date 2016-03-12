'''
	run_mlp takes size of hidden layer 
'''

import tensorflow as tf
import numpy as np
import input_data
import sys

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_hs, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_hs[0]))
    for w_h in w_hs[1:]:
    	h = tf.nn.sigmoid(tf.matmul(h, w_h))
    return tf.matmul(h, w_o)

def run_mlp(input_dim = 784, output_dim = 10, hidden_weights = [12], lr = 0.001, num_iter = 3, 
	saved_model_path = "model.ckpt", mode = "train", output_file = "out_file"):
	
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	
	trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

	X = tf.placeholder("float", [None, input_dim])
	Y = tf.placeholder("float", [None, output_dim])

	w_hs = []
	w_hs.append(init_weights([input_dim, hidden_weights[0]]))
	for i in xrange(len(hidden_weights)-1):
		w_hs.append(init_weights([hidden_weights[i], hidden_weights[i+1]]))

	w_o = init_weights([hidden_weights[-1], output_dim])

	py_x = model(X, w_hs, w_o)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
	# construct an optimizer, choice of learning rate
	train_op = tf.train.RMSPropOptimizer(lr, 0.9).minimize(cost)
	predict_op = tf.argmax(py_x, 1)

	saver = tf.train.Saver()

	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)

	with open('output.txt','w') as out:
		sys.stdout = out
		if mode == "train":
			for i in range(num_iter):
			    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
			        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
			    print i, np.mean(np.argmax(teY, axis=1) ==
			                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))

			saved = saver.save(sess, saved_model_path)
		elif mode == "test":
			saver.restore(sess, saved_model_path)
			for i in [1,2,3]:
			    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
			        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
			    print i, np.mean(np.argmax(teY, axis=1) ==
			                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))
		else:
			fsock = open('error.log', 'w')
			sys.stderr = fsock
			raise ValueError('Unidentified Option!')

def main():
	sys_out = sys.stdout
	hidden_weights = [200, 35, 23]
	run_mlp(784, 10, hidden_weights, 0.005, mode = "ttest")

if __name__ == "__main__":
	main()
