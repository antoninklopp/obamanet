from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Lambda, TimeDistributed
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import pickle as pkl
from keras.callbacks import TensorBoard
from time import time
import tensorflow as tf

#########################################################################################

time_delay = 50 #0
look_back = 50
n_epoch = 50
n_videos = 50
tbCallback = TensorBoard(log_dir="logs/{}".format(time())) # TensorBoard(log_dir='./Graph', histogram_freq=0, batch_size=n_batch, write_graph=True, write_images=True)

#########################################################################################

class Arguments:

	def __init__(self, keep_prob):
		self.keep_prob = keep_prob
		self.num_layers = 0
		self.batch_size = 0
		self.seq_length = 0
		self.rnn_size = 0
		self.grad_clip = 0

class Model:

	dimin = 20
	dimout = 20

	def standardL2Model(self, infer=False):

		args = Arguments(keep_prob=False)
		args.keep_prob = 0.5
		
		args.batch_size = 1

		args.seq_length = 25
		args.rnn_size = 20

		args.grad_clip = True
	
		cell_fn = tf.nn.rnn_cell.LSTMCell
		cell = cell_fn(args.rnn_size, state_is_tuple=True)
	
		if infer == False and args.keep_prob < 1: # training mode
		  cell0 = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = args.keep_prob)
		  cell1 = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = args.keep_prob, output_keep_prob = args.keep_prob)
		  self.network = tf.nn.rnn_cell.MultiRNNCell([cell0] * (args.num_layers -1) + [cell1], state_is_tuple=True)
		else:
		  self.network = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
	
	
		self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dimin])
		self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dimout])
		self.initial_state = self.network.zero_state(batch_size=args.batch_size, dtype=tf.float32)
	
		with tf.variable_scope('rnnlm'):
		  output_w = tf.get_variable("output_w", [args.rnn_size, self.dimout])
		  output_b = tf.get_variable("output_b", [self.dimout])
	
		inputs = tf.split(self.input_data, 1, args.seq_length)
		inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
	
		outputs, states = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, self.network, loop_function=None, scope='rnnlm')
	
		output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
		output = tf.nn.xw_plus_b(output, output_w, output_b)
		self.final_state = states
		self.output = output
	
		flat_target_data = tf.reshape(self.target_data,[-1, self.dimout])
			
		lossfunc = tf.reduce_sum(tf.squared_difference(flat_target_data, output))
		#lossfunc = tf.reduce_sum(tf.abs(flat_target_data - output))
		self.cost = lossfunc / (args.batch_size * args.seq_length * self.dimout)
	
		self.lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

def get_data():
	# Load the files
	with open('data/audio_kp/audio_kp2748_mel.pickle', 'rb') as pkl_file:
		audio_kp = pkl.load(pkl_file)
	with open('data/pca/pkp2748.pickle', 'rb') as pkl_file:
		video_kp = pkl.load(pkl_file)
	with open('data/pca/pca2748.pickle', 'rb') as pkl_file:
		pca = pkl.load(pkl_file)

	# Get the data

	X, y = [], [] # Create the empty lists
	# Get the common keys
	keys_audio = audio_kp.keys()
	keys_video = video_kp.keys()
	keys = sorted(list(set(keys_audio).intersection(set(keys_video))))

	for key in tqdm(keys[0:n_videos]):
		print(key)
		audio = audio_kp[key]
		video = video_kp[key]
		print(len(audio))
		if (len(audio) - len(video) > 1):
			continue
		if (len(audio) > len(video)):
			audio = audio[0:len(video)]
		else:
			video = video[0:len(audio)]
		start = (time_delay-look_back) if (time_delay-look_back > 0) else 0
		for i in range(start, len(audio)-look_back):
			a = np.array(audio[i:i+look_back])
			v = np.array(video[i+look_back-time_delay]).reshape((1, -1))
			X.append(a)
			y.append(v)

	X = np.array(X)
	y = np.array(y)
	shapeX = X.shape
	shapey = y.shape
	print('Shapes:', X.shape, y.shape)
	X = X.reshape(-1, X.shape[2])
	y = y.reshape(-1, y.shape[2])
	print('Shapes:', X.shape, y.shape)

	scalerX = MinMaxScaler(feature_range=(0, 1))
	scalery = MinMaxScaler(feature_range=(0, 1))

	X = scalerX.fit_transform(X)
	y = scalery.fit_transform(y)


	X = X.reshape(shapeX)
	y = y.reshape(shapey[0], shapey[2])

	print('Shapes:', X.shape, y.shape)
	print('X mean:', np.mean(X), 'X var:', np.var(X))
	print('y mean:', np.mean(y), 'y var:', np.var(y))

	split1 = int(0.8*X.shape[0])
	split2 = int(0.9*X.shape[0])

	train_x = X[0:split1]
	train_y = y[0:split1]
	val_x = X[split1:split2]
	val_y = y[split1:split2]
	test_x = X[split2:]
	test_y = y[split2:]

	return train_x, train_y, val_x, val_y, test_x, test_y

if __name__ == "__main__":
	train_X, train_y, val_X, val_y, test_X, test_y = get_data()

	# Initialize the model
	model = Model().standardL2Model()
	print(model.summary())

	# train LSTM with validation data
	model.fit(train_X, train_y, epochs=5, batch_size=16, 
		verbose=1, shuffle=True, callbacks=[tbCallback], validation_data=(val_X, val_y))
	# model.reset_states()
	test_error = np.mean(np.square(test_y - model.predict(test_X)))
	# model.reset_states()
	print('Test Error: ', test_error)

	# Save the model
	model.save('my_model.h5')
	model.save_weights('my_model_weights.h5')
	print('Saved Model.')



