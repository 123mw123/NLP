import numpy as np
from keras.models import model_from_json
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
config.log_device_placement = True 
sess = tf.Session(config=config)
set_session(sess)


def init():
	json_file = open('./model/lstm.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("./model/lstm.h5")
	loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print("Loaded model from disk")
	graph = tf.get_default_graph()
	return loaded_model,graph