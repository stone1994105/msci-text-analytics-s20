import numpy as np
import pandas as pd
import os
import sys
from gensim.models import Word2Vec
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from keras.layers import Dense, Embedding, Dropout, Activation, Flatten
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.models import Sequential

def load_data(file_path):
	pos_train = pd.read_csv(os.path.join(file_path + "\\pos_train.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	neg_train = pd.read_csv(os.path.join(file_path + "\\neg_train.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	pos_test = pd.read_csv(os.path.join(file_path + "\\pos_test.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	neg_test = pd.read_csv(os.path.join(file_path + "\\neg_test.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	pos_vali = pd.read_csv(os.path.join(file_path + "\\pos_vali.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	neg_vali = pd.read_csv(os.path.join(file_path + "\\neg_vali.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	return pos_train,neg_train,pos_test,neg_test,pos_vali,neg_vali

def main(file_path):
	pos_train,neg_train,pos_test,neg_test,pos_vali,neg_vali = load_data(file_path) 

	pos_train['target'] = 1
	neg_train['target'] = 0
	train = pd.concat([pos_train,neg_train])
	train = shuffle(train, random_state=0).reset_index(drop=True)

	pos_test['target'] = 1
	neg_test['target'] = 0
	test = pd.concat([pos_test,neg_test])
	test = shuffle(test, random_state=0).reset_index(drop=True)

	pos_vali['target'] = 1
	neg_vali['target'] = 0
	vali = pd.concat([pos_vali,neg_vali])
	vali = shuffle(vali, random_state=0).reset_index(drop=True)

	x_train = train['Review'].to_numpy()
	y_train = train['target'].to_numpy()
	x_test = test['Review'].to_numpy()
	y_test = test['target'].to_numpy()
	x_vali = vali['Review'].to_numpy()
	y_vali = vali['target'].to_numpy()

	tokenizer_obj = Tokenizer()
	tokenizer_obj.fit_on_texts(pd.concat([train,test,vali]))

	x_train_sequences = tokenizer_obj.texts_to_sequences(x_train)
	x_test_sequences = tokenizer_obj.texts_to_sequences(x_test)
	x_vali_sequences = tokenizer_obj.texts_to_sequences(x_vali)

	max_length_arr = [len(s) for s in (x_train_sequences + x_test_sequences + x_vali_sequences)]
	max_length = max(max_length_arr)

	x_train_padded = pad_sequences(x_train_sequences, maxlen=max_length, padding='post', truncating='post')
	x_test_padded = pad_sequences(x_test_sequences, maxlen=max_length, padding='post', truncating='post')
	x_vali_padded = pad_sequences(x_vali_sequences, maxlen=max_length, padding='post', truncating='post')

	y_train = to_categorical(np.asarray(y_train))
	y_test = to_categorical(np.asarray(y_test))
	y_vali = to_categorical(np.asarray(y_vali))

	w2v_model = Word2Vec.load("C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 3\\data\\w2v.model")
	pretrained_weights = w2v_model.wv.vectors
	vocab_size, emdedding_size = pretrained_weights.shape

	model = Sequential()
	model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	model.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.001)))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))

	model.summary()
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	model.fit(x_train_padded, y_train, batch_size=1024, epochs=3, validation_data=(x_vali_padded, y_vali))

	score, acc = model.evaluate(x_test_padded, y_test, batch_size=1024)
	print("Accuracy on the test set = {0:4.3f}".format(acc))

if __name__ == "__main__":
	main(sys.argv[1])