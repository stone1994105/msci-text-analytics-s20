import numpy as np
import pandas as pd
import os
import sys
import pickle
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
	# load data
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

	doc = pd.concat([train,test,vali])

	# Vectorize text data
	tokenizer_obj = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
	tokenizer_obj.fit_on_texts(doc['Review'].to_numpy())

	# save tokenizer_obj
	with open('tokenizer.pickle', 'wb') as f:
		pickle.dump(tokenizer_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

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

	# Load Word2Vec embedding

	w2v_model = Word2Vec.load("C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 3\\data\\w2v.model")
	pretrained_weights = w2v_model.wv.vectors
	vocab_size, emdedding_size = pretrained_weights.shape

	# Build the models

	# model 1. No regularization,No dropout,sigmoid

	# model1 = Sequential()
	# model1.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model1.add(Dense(64, activation='sigmoid'))
	# model1.add(Flatten())
	# model1.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model1.summary()
	# model1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model1.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model1.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,no dropout,sigmoid) = {0:4.3f}".format(acc))

	# # model 2. No regularization,No dropout,ReLU

	# model2 = Sequential()
	# model2.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model2.add(Dense(64, activation='relu'))
	# model2.add(Flatten())
	# model2.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model2.summary()
	# model2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model2.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model2.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,no dropout,ReLU) = {0:4.3f}".format(acc))

	# # model 3. No regularization,No dropout,tanh

	# model3 = Sequential()
	# model3.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model3.add(Dense(64, activation='tanh'))
	# model3.add(Flatten())
	# model3.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model3.summary()
	# model3.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model3.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model3.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,no dropout,tanh) = {0:4.3f}".format(acc))

	# # model 4. L2 regularization = 0.01,No dropout,sigmoid

	# model4 = Sequential()
	# model4.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model4.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.01)))
	# model4.add(Flatten())
	# model4.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model4.summary()
	# model4.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model4.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model4.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2 regularization = 0.01,No dropout,sigmoid) = {0:4.3f}".format(acc))

	# # model 5. L2 regularization = 0.01,No dropout,ReLU

	# model5 = Sequential()
	# model5.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model5.add(Dense(64, activation='relu', kernel_regularizer= l2(0.01)))
	# model5.add(Flatten())
	# model5.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model5.summary()
	# model5.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model5.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model5.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2 regularization = 0.01,No dropout,ReLU) = {0:4.3f}".format(acc))
	# model5.save("C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 4\\data\\nn_relu.model")

	# # model 6. L2 regularization = 0.01,No dropout,tanh

	# model6 = Sequential()
	# model6.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model6.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.01)))
	# model6.add(Flatten())
	# model6.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model6.summary()
	# model6.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model6.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model6.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2 regularization = 0.01,No dropout,tanh) = {0:4.3f}".format(acc))

	# # model 7. L2-regularization = 0.01, Dropout = 0.3,sigmoid

	# model7 = Sequential()
	# model7.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model7.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.01)))
	# model7.add(Flatten())
	# model7.add(Dropout(0.3))
	# model7.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model7.summary()
	# model7.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model7.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model7.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.3,sigmoid) = {0:4.3f}".format(acc))

	# # model 8. L2-regularization = 0.01, Dropout = 0.3,ReLU

	# model8 = Sequential()
	# model8.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model8.add(Dense(64, activation='relu', kernel_regularizer= l2(0.01)))
	# model8.add(Flatten())
	# model8.add(Dropout(0.3))
	# model8.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model8.summary()
	# model8.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model8.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model8.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.3,ReLU) = {0:4.3f}".format(acc))

	# # model 9. L2-regularization = 0.01, Dropout = 0.3,tanh

	# model9 = Sequential()
	# model9.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model9.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.01)))
	# model9.add(Flatten())
	# model9.add(Dropout(0.3))
	# model9.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model9.summary()
	# model9.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model9.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model9.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.3,tanh) = {0:4.3f}".format(acc))

	# # model 10. L2-regularization = 0.01, Dropout = 0.4,sigmoid

	# model10 = Sequential()
	# model10.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model10.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.01)))
	# model10.add(Flatten())
	# model10.add(Dropout(0.4))
	# model10.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model10.summary()
	# model10.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model10.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model10.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.4,sigmoid) = {0:4.3f}".format(acc))

	# # model 11. L2-regularization = 0.01, Dropout = 0.4,ReLU

	# model11 = Sequential()
	# model11.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model11.add(Dense(64, activation='relu', kernel_regularizer= l2(0.01)))
	# model11.add(Flatten())
	# model11.add(Dropout(0.4))
	# model11.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model11.summary()
	# model11.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model11.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model11.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.4,ReLU) = {0:4.3f}".format(acc))

	# # model 12. L2-regularization = 0.01, Dropout = 0.4,tanh

	# model12 = Sequential()
	# model12.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model12.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.01)))
	# model12.add(Flatten())
	# model12.add(Dropout(0.4))
	# model12.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model12.summary()
	# model12.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model12.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model12.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.4,tanh) = {0:4.3f}".format(acc))

	# # model 13. L2-regularization = 0.01, Dropout = 0.5,sigmoid

	# model13 = Sequential()
	# model13.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model13.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.01)))
	# model13.add(Flatten())
	# model13.add(Dropout(0.5))
	# model13.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model13.summary()
	# model13.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model13.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model13.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.5,sigmoid) = {0:4.3f}".format(acc))

	# # model 14. L2-regularization = 0.01, Dropout = 0.5,ReLU

	# model14 = Sequential()
	# model14.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model14.add(Dense(64, activation='relu', kernel_regularizer= l2(0.01)))
	# model14.add(Flatten())
	# model14.add(Dropout(0.5))
	# model14.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model14.summary()
	# model14.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model14.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model14.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.5,ReLU) = {0:4.3f}".format(acc))

	# # model 15. L2-regularization = 0.01, Dropout = 0.5,tanh

	# model15 = Sequential()
	# model15.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model15.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.01)))
	# model15.add(Flatten())
	# model15.add(Dropout(0.5))
	# model15.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model15.summary()
	# model15.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model15.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model15.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.5,tanh) = {0:4.3f}".format(acc))
	# model15.save("C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 4\\data\\nn_tanh.model")

	# # model 16. L2-regularization = 0.01, Dropout = 0.6,sigmoid
	# model16 = Sequential()
	# model16.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model16.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.01)))
	# model16.add(Flatten())
	# model16.add(Dropout(0.6))
	# model16.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model16.summary()
	# model16.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model16.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model16.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.6,sigmoid) = {0:4.3f}".format(acc))

	# # model 17. L2-regularization = 0.01, Dropout = 0.6,ReLU

	# model17 = Sequential()
	# model17.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model17.add(Dense(64, activation='relu', kernel_regularizer= l2(0.01)))
	# model17.add(Flatten())
	# model17.add(Dropout(0.6))
	# model17.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model17.summary()
	# model17.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model17.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model17.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.6,ReLU) = {0:4.3f}".format(acc))

	# # model 18. L2-regularization = 0.01, Dropout = 0.6,tanh

	# model18 = Sequential()
	# model18.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model18.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.01)))
	# model18.add(Flatten())
	# model18.add(Dropout(0.6))
	# model18.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.01), name='output_layer'))
	# model18.summary()
	# model18.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model18.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model18.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.01, Dropout = 0.6,tanh) = {0:4.3f}".format(acc))

	# # model 19. L2 regularization = 0.001,No dropout,sigmoid

	# model19 = Sequential()
	# model19.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model19.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.001)))
	# model19.add(Flatten())
	# model19.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model19.summary()
	# model19.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model19.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model19.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2 regularization = 0.001,No dropout,sigmoid) = {0:4.3f}".format(acc))

	# # model 20. L2 regularization = 0.001,No dropout,ReLU

	# model20 = Sequential()
	# model20.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model20.add(Dense(64, activation='relu', kernel_regularizer= l2(0.001)))
	# model20.add(Flatten())
	# model20.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model20.summary()
	# model20.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model20.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model20.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2 regularization = 0.001,No dropout,ReLU) = {0:4.3f}".format(acc))

	# # model 21. L2 regularization = 0.001,No dropout,tanh

	# model21 = Sequential()
	# model21.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model21.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.001)))
	# model21.add(Flatten())
	# model21.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model21.summary()
	# model21.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model21.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model21.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2 regularization = 0.001,No dropout,tanh) = {0:4.3f}".format(acc))

	# # model 22. L2-regularization = 0.001, Dropout = 0.3,sigmoid

	# model22 = Sequential()
	# model22.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model22.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.001)))
	# model22.add(Flatten())
	# model22.add(Dropout(0.3))
	# model22.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model22.summary()
	# model22.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model22.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model22.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.3,sigmoid) = {0:4.3f}".format(acc))

	# # model 23. L2-regularization = 0.001, Dropout = 0.3,ReLU

	# model23 = Sequential()
	# model23.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model23.add(Dense(64, activation='relu', kernel_regularizer= l2(0.001)))
	# model23.add(Flatten())
	# model23.add(Dropout(0.3))
	# model23.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model23.summary()
	# model23.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model23.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model23.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.3,ReLU) = {0:4.3f}".format(acc))

	# # model 24. L2-regularization = 0.001, Dropout = 0.3,tanh

	# model24 = Sequential()
	# model24.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model24.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.001)))
	# model24.add(Flatten())
	# model24.add(Dropout(0.3))
	# model24.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model24.summary()
	# model24.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model24.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model24.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.3,tanh) = {0:4.3f}".format(acc))

	# # model 25. L2-regularization = 0.001, Dropout = 0.4,sigmoid

	# model25 = Sequential()
	# model25.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model25.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.001)))
	# model25.add(Flatten())
	# model25.add(Dropout(0.4))
	# model25.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model25.summary()
	# model25.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model25.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model25.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.4,sigmoid) = {0:4.3f}".format(acc))
	# model25.save('C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 4\\data\\nn_sigmoid.model')

	# # model 26. L2-regularization = 0.001, Dropout = 0.4,ReLU

	# model26 = Sequential()
	# model26.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model26.add(Dense(64, activation='relu', kernel_regularizer= l2(0.001)))
	# model26.add(Flatten())
	# model26.add(Dropout(0.4))
	# model26.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model26.summary()
	# model26.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model26.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model26.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.4,ReLU) = {0:4.3f}".format(acc))

	# # model 27. L2-regularization = 0.001, Dropout = 0.4,tanh

	# model27 = Sequential()
	# model27.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model27.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.001)))
	# model27.add(Flatten())
	# model27.add(Dropout(0.4))
	# model27.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model27.summary()
	# model27.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model27.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model27.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.4,tanh) = {0:4.3f}".format(acc))

	# # model 28. L2-regularization = 0.001, Dropout = 0.5,sigmoid

	# model28 = Sequential()
	# model28.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model28.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.001)))
	# model28.add(Flatten())
	# model28.add(Dropout(0.5))
	# model28.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model28.summary()
	# model28.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model28.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model28.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.5,sigmoid) = {0:4.3f}".format(acc))

	# # model 29. L2-regularization = 0.001, Dropout = 0.5,ReLU

	# model29 = Sequential()
	# model29.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model29.add(Dense(64, activation='relu', kernel_regularizer= l2(0.001)))
	# model29.add(Flatten())
	# model29.add(Dropout(0.5))
	# model29.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model29.summary()
	# model29.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model29.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model29.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.5,ReLU) = {0:4.3f}".format(acc))

	# # model 30. L2-regularization = 0.001, Dropout = 0.5,tanh

	# model30 = Sequential()
	# model30.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model30.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.001)))
	# model30.add(Flatten())
	# model30.add(Dropout(0.5))
	# model30.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model30.summary()
	# model30.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model30.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model30.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.5,tanh) = {0:4.3f}".format(acc))

	# # model 31. L2-regularization = 0.001, Dropout = 0.6,sigmoid
	# model31 = Sequential()
	# model31.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model31.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.001)))
	# model31.add(Flatten())
	# model31.add(Dropout(0.6))
	# model31.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model31.summary()
	# model31.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model31.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model31.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.6,sigmoid) = {0:4.3f}".format(acc))

	# # model 32. L2-regularization = 0.001, Dropout = 0.6,ReLU

	# model32 = Sequential()
	# model32.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model32.add(Dense(64, activation='relu', kernel_regularizer= l2(0.001)))
	# model32.add(Flatten())
	# model32.add(Dropout(0.6))
	# model32.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model32.summary()
	# model32.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model32.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model32.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.6,ReLU) = {0:4.3f}".format(acc))

	# # model 33. L2-regularization = 0.01, Dropout = 0.6,tanh

	# model33 = Sequential()
	# model33.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model33.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.001)))
	# model33.add(Flatten())
	# model33.add(Dropout(0.6))
	# model33.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
	# model33.summary()
	# model33.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model33.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model33.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (L2-regularization = 0.001, Dropout = 0.6,tanh) = {0:4.3f}".format(acc))

	# # model 34. No regularization, Dropout = 0.3, sigmoid

	# model34 = Sequential()
	# model34.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model34.add(Dense(64, activation='sigmoid'))
	# model34.add(Flatten())
	# model34.add(Dropout(0.3))
	# model34.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model34.summary()
	# model34.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model34.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model34.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.3,sigmoid) = {0:4.3f}".format(acc))

	# # model 35. No regularization, Dropout = 0.3, ReLU

	# model35 = Sequential()
	# model35.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model35.add(Dense(64, activation='relu'))
	# model35.add(Flatten())
	# model35.add(Dropout(0.3))
	# model35.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model35.summary()
	# model35.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model35.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model35.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.3,ReLU) = {0:4.3f}".format(acc))

	# # model 36. No regularization, Dropout = 0.3, tanh

	# model36 = Sequential()
	# model36.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model36.add(Dense(64, activation='tanh'))
	# model36.add(Flatten())
	# model36.add(Dropout(0.3))
	# model36.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model36.summary()
	# model36.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model36.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model36.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.3,tanh) = {0:4.3f}".format(acc))

	# # model 37. No regularization, Dropout = 0.4, sigmoid

	# model37 = Sequential()
	# model37.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model37.add(Dense(64, activation='sigmoid'))
	# model37.add(Flatten())
	# model37.add(Dropout(0.4))
	# model37.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model37.summary()
	# model37.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model37.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model37.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.4,sigmoid) = {0:4.3f}".format(acc))

	# # model 38. No regularization, Dropout = 0.4, ReLU

	# model38 = Sequential()
	# model38.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model38.add(Dense(64, activation='relu'))
	# model38.add(Flatten())
	# model38.add(Dropout(0.4))
	# model38.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model38.summary()
	# model38.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model38.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model38.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.4,ReLU) = {0:4.3f}".format(acc))

	# # model 39. No regularization, Dropout = 0.4, tanh

	# model39 = Sequential()
	# model39.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model39.add(Dense(64, activation='tanh'))
	# model39.add(Flatten())
	# model39.add(Dropout(0.4))
	# model39.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model39.summary()
	# model39.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model39.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model36.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.4,tanh) = {0:4.3f}".format(acc))

	# # model 40. No regularization, Dropout = 0.5, sigmoid

	# model40 = Sequential()
	# model40.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model40.add(Dense(64, activation='sigmoid'))
	# model40.add(Flatten())
	# model40.add(Dropout(0.5))
	# model40.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model40.summary()
	# model40.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model40.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model40.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.5,sigmoid) = {0:4.3f}".format(acc))

	# # model 41. No regularization, Dropout = 0.5, ReLU

	# model41 = Sequential()
	# model41.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model41.add(Dense(64, activation='relu'))
	# model41.add(Flatten())
	# model41.add(Dropout(0.5))
	# model41.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model41.summary()
	# model41.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model41.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model41.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.5,ReLU) = {0:4.3f}".format(acc))

	# # model 42. No regularization, Dropout = 0.5, tanh

	# model42 = Sequential()
	# model42.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model42.add(Dense(64, activation='tanh'))
	# model42.add(Flatten())
	# model42.add(Dropout(0.5))
	# model42.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model42.summary()
	# model42.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model42.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model42.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.5,tanh) = {0:4.3f}".format(acc))

	# # model 43. No regularization, Dropout = 0.6, sigmoid

	# model43 = Sequential()
	# model43.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model43.add(Dense(64, activation='sigmoid'))
	# model43.add(Flatten())
	# model43.add(Dropout(0.6))
	# model43.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model43.summary()
	# model43.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model43.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model43.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.6,sigmoid) = {0:4.3f}".format(acc))

	# # model 44. No regularization, Dropout = 0.6, ReLU

	# model44 = Sequential()
	# model44.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model44.add(Dense(64, activation='relu'))
	# model44.add(Flatten())
	# model44.add(Dropout(0.6))
	# model44.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model44.summary()
	# model44.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model44.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model44.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.6,ReLU) = {0:4.3f}".format(acc))

	# # model 45. No regularization, Dropout = 0.6, tanh

	# model45 = Sequential()
	# model45.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=max_length, weights=[pretrained_weights]))
	# model45.add(Dense(64, activation='tanh'))
	# model45.add(Flatten())
	# model45.add(Dropout(0.6))
	# model45.add(Dense(2, activation = 'softmax', name='output_layer'))
	# model45.summary()
	# model45.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# model45.fit(x_train_padded, y_train, batch_size=1024, epochs=10, validation_data=(x_vali_padded, y_vali))
	# score, acc = model45.evaluate(x_test_padded, y_test, batch_size=1024)
	# print("Accuracy (no regularization,Dropout = 0.6,tanh) = {0:4.3f}".format(acc))



if __name__ == "__main__":
	main(sys.argv[1])