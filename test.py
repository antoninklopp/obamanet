import tensorflow as tf
from train import get_data
from keras.models import load_model
import numpy as np

if __name__ == "__main__":
    model = load_model('my_model.h5')
    train_X, train_y, val_X, val_y, test_X, test_y = get_data()
    loss, accuracy = model.evaluate(test_X, test_y)
    print(loss, accuracy)
    #for value, prediction in zip(test_y, model.predict(test_X)):
    #    print(value, prediction, np.mean(np.square(value - prediction)))