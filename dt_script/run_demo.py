'''
	run_mlp:
	when load a model, need to specify the shape of model
'''

import tensorflow as tf
import numpy as np
import input_data
import sys, pickle

class INPUT_FLAG:
	def __init__(self):
		self.input_dim, self.trX, self.trY, self.teX, self.teY = \
		None, None, None, None, None

def update_data_flag(
	input_flag, 
	train_dir = "", 
	test_dir = "", 
	opt = "", 
	output_dim = 2,
	mode = "train"):

	if opt == "mnist" or opt == "MNIST":
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		input_flag.trX, input_flag.trY, input_flag.teX, input_flag.teY = \
		mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	else:
		if mode == "train":
			train_data = np.loadtxt(open(train_dir,"rb"), delimiter=",", dtype=int)
			input_flag.trX, input_flag.trY = train_data[:, :-1], train_data[:, -1]
			temp_tr = np.zeros((len(input_flag.trY), output_dim))
			temp_tr[np.arange(len(input_flag.trY)), input_flag.trY] = 1	
			input_flag.trY = temp_tr
			input_flag.teX, input_flag.teY = input_flag.trX, input_flag.trY
		elif mode == "test":	
			test_data = np.loadtxt(open(test_dir,"rb"), delimiter=",", dtype=int)
			input_flag.teX, input_flag.teY = test_data[:, :-1], test_data[:, -1]
			temp_te = np.zeros((len(input_flag.teY), output_dim))
			temp_te[np.arange(len(input_flag.teY)), input_flag.teY] = 1
			input_flag.teY = temp_te
	input_flag.input_dim = np.size(input_flag.teX, 1)


def test_update():
	input_flag = INPUT_FLAG()
	mode = "train"
	update_data_flag(input_flag, train_dir = "sample_train.txt", output_dim = 2, mode = mode)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_hs, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_hs[0]))
    for w_h in w_hs[1:]:
    	h = tf.nn.sigmoid(tf.matmul(h, w_h))
    return tf.matmul(h, w_o)


'''
	warning: when run XOR experiment, num_iter should be large
'''
def run_mlp(
	hidden_weights = [12], 
	lr = 0.002, 
	num_iter = 5, 
	train_dir = "", 
	test_dir = "",
	output_dim = 2, 
	saved_model_path = "model.ckpt",
	saved_weights_path = "weights.pckl",
	mode = "train", 
	output_file = "out_file", 
	opt = "user_data"):

	input_flag = INPUT_FLAG()
	if opt == "mnist":
		output_dim = 10
		update_data_flag(input_flag, "", "", opt, output_dim)
	else:
		update_data_flag(input_flag, train_dir, test_dir, output_dim, mode = mode)
	
	trX, trY, teX, teY, input_dim = input_flag.trX, input_flag.trY, \
	input_flag.teX, input_flag.teY, input_flag.input_dim
	
	X = tf.placeholder("float", [None, input_dim])
	Y = tf.placeholder("float", [None, output_dim])
	
	# if mode == "train":
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
	sess.run(tf.initialize_all_variables())

	with open(output_file,'w') as out:
		sys.stdout = out
		if mode == "test":
			saver.restore(sess, saved_model_path)
			print np.mean(np.argmax(teY, axis=1) == \
				sess.run(predict_op, feed_dict={X: teX, Y: teY}))
		elif mode == "train":
			for i in range(num_iter):
				if opt == "mnist" or opt == "MNIST":
					for start, end in zip(range(0, len(trX), 50), range(50, len(trX), 50)):\
					sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
				else:
					sess.run(train_op, feed_dict={X: trX, Y: trY})
				
				print i, np.mean(np.argmax(trY, axis=1) == \
					sess.run(predict_op, feed_dict={X: trX, Y: trY}))
			# save session
			saver.save(sess, saved_model_path)   
			# save weights as pickle
			weights = sess.run(w_hs)
			weights.append(sess.run(w_o))
			with open(saved_weights_path, 'w') as f:
				pickle.dump(weights, f)
		else:
			fsock = open('error.log', 'w')
			sys.stderr = fsock
			raise ValueError('Unidentified Option!')

def main():
	# # MNIST
	# hidden_weights = [300, 65, 20]
	# # train / test
	# run_mlp(
	# 	hidden_weights, 
	# 	num_iter = 5, 
	# 	mode = "test", 
	# 	opt = "mnist", 
	# 	saved_model_path = "model.ckpt",
	# output_file = "output.txt")

	# XOR
	hidden_weights = [6]
	
	# # train
	# run_mlp(
	# 	hidden_weights, 
	# 	num_iter = 10000, 
	# 	train_dir = "sample_train.txt", 
	# 	output_dim = 2, 
	# 	mode = "train", 
	# 	saved_model_path = "model.ckpt", 
	# 	saved_weights_path = "weights.pckl",
	# 	output_file = "output.txt"
	# 	)

	# test
	run_mlp(
		hidden_weights,
		test_dir = "sample_train.txt", 
		saved_model_path = "model.ckpt", 
		output_dim = 2, 
		mode = "test",
		output_file = "output.txt"
		)

if __name__ == "__main__":
	main()
