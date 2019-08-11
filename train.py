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

#########################################################################################

time_delay = 20 #0
look_back = 50
n_epoch = 50
n_videos = 200
tbCallback = TensorBoard(log_dir="logs/{}".format(time())) # TensorBoard(log_dir='./Graph', histogram_freq=0, batch_size=n_batch, write_graph=True, write_images=True)

#########################################################################################

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
		audio = audio_kp[key]
		video = video_kp[key]
		print(len(audio), len(video))
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
	model = Sequential()
	model.add(LSTM(25, input_shape=(look_back, 26)))
	model.add(Dense(8))
	model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
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



