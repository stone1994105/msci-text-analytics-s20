import sys
import os
from tensorflow import keras
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

def main(file_path,classifier):
	# load tokenizer
	with open('tokenizer.pickle', 'rb') as f:
		loaded_tokenizer = pickle.load(f)

	# load model
	script_dir = os.path.dirname(__file__)

	if classifier == 'relu':
		rel_path = "data\\nn_relu.model"
	elif classifier == 'sigmoid':
		rel_path = "data\\nn_sigmoid.model"
	elif classifier == 'tanh':
		rel_path = "data\\nn_tanh.model"

	classifier_path = os.path.join(script_dir, rel_path)

	model = keras.models.load_model(classifier_path)

	# load txt file
	with open (file_path) as f:
		lines = f.read().lower().splitlines()
	lines = np.array(lines)

	seq = loaded_tokenizer.texts_to_sequences(lines)
	padded = pad_sequences(seq,maxlen=94)
	pred = model.predict(padded)
	print("probability:\n",pred)
	pred_class = np.argmax(model.predict(padded), axis = -1)
	print("class:\n",pred_class)


if __name__ == "__main__":
	main(sys.argv[1],sys.argv[2])