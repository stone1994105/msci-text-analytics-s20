import os
import sys
import pickle
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(file_path):
	pos_train = pd.read_csv(os.path.join(file_path + "\\pos_train.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	neg_train = pd.read_csv(os.path.join(file_path + "\\neg_train.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	pos_test = pd.read_csv(os.path.join(file_path + "\\pos_test.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	neg_test = pd.read_csv(os.path.join(file_path + "\\neg_test.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	pos_vali = pd.read_csv(os.path.join(file_path + "\\pos_vali.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	neg_vali = pd.read_csv(os.path.join(file_path + "\\neg_vali.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	pos_train_ns = pd.read_csv(os.path.join(file_path + "\\pos_train_ns.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	neg_train_ns = pd.read_csv(os.path.join(file_path + "\\neg_train_ns.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	pos_test_ns = pd.read_csv(os.path.join(file_path + "\\pos_test_ns.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	neg_test_ns = pd.read_csv(os.path.join(file_path + "\\neg_test_ns.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	pos_vali_ns = pd.read_csv(os.path.join(file_path + "\\pos_vali_ns.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	neg_vali_ns = pd.read_csv(os.path.join(file_path + "\\neg_vali_ns.csv"), sep = 'delimiter', names = ['Review'], engine = 'python')
	return pos_train,neg_train,pos_test,neg_test,pos_vali,neg_vali,pos_train_ns,neg_train_ns,pos_test_ns,neg_test_ns,pos_vali_ns,neg_vali_ns

def train_data(x, y, count_vect, alpha):
	x_count = count_vect.fit_transform(x)
	tfidf = TfidfTransformer()
	x_tfidf = tfidf.fit_transform(x_count)
	clf = MultinomialNB(alpha = alpha).fit(x_tfidf, y)
	return clf, count_vect, tfidf

def evaluate(x, y, clf, count_vect, tfidf):
	x_count = count_vect.transform(x)
	x_tfidf = tfidf.transform(x_count)
	preds = clf.predict(x_tfidf)
	return {
		'accuracy': accuracy_score(y, preds),
		'precision': precision_score(y, preds),
		'recall': recall_score(y, preds),
		'f1': f1_score(y, preds),
		}

def alpha_tuning(x_train,y_train,x_vali,y_vali,count_vect,alpha_list):
	accuracy = []
	for alpha in alpha_list:
		clf, count_vect, tfidf = train_data(x_train, y_train, count_vect, alpha)
		x_count = count_vect.transform(x_vali)
		x_tfidf = tfidf.transform(x_count)
		preds = clf.predict(x_tfidf)
		accuracy.append(accuracy_score(y_vali, preds))
	index = np.argmax(accuracy)
	best_alpha = alpha_list[index]
	return best_alpha

# Build CountVectorizer for unigram, bigram, and unigram + bigram
vect_unigram = CountVectorizer(analyzer='word', ngram_range = (1, 1))
vect_bigram = CountVectorizer(analyzer='word', ngram_range = (2, 2))
vect_unibigram = CountVectorizer(analyzer='word', ngram_range = (1, 2))

# alpha values to test on validation set
alpha_list = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,2,3]

def main(file_path):
	test_scores = {}
	test_scores_optimized = {}
	# Load Data
	# file_path = sys.argv[1]
	pos_train,neg_train,pos_test,neg_test,pos_vali,neg_vali,\
	pos_train_ns,neg_train_ns,pos_test_ns,neg_test_ns,pos_vali_ns,neg_vali_ns = load_data(file_path) 

	# Concat positive and negative data with labels
	# 1 for positive, 0 for negative
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

	pos_train_ns['target'] = 1
	neg_train_ns['target'] = 0
	train_ns = pd.concat([pos_train_ns,neg_train_ns])
	train_ns = shuffle(train_ns, random_state=0).reset_index(drop=True)

	pos_test_ns['target'] = 1
	neg_test_ns['target'] = 0
	test_ns = pd.concat([pos_test_ns,neg_test_ns])
	test_ns = shuffle(test_ns, random_state=0).reset_index(drop=True)

	pos_vali_ns['target'] = 1
	neg_vali_ns['target'] = 0
	vali_ns = pd.concat([pos_vali_ns,neg_vali_ns])
	vali_ns = shuffle(vali_ns, random_state=0).reset_index(drop=True)

	# Prepare x and y
	x_train = train['Review']
	y_train = train['target']
	x_test = test['Review']
	y_test = test['target']
	x_vali = vali['Review']
	y_vali = vali['target']

	x_train_ns = train_ns['Review']
	y_train_ns = train_ns['target']
	x_test_ns = test_ns['Review']
	y_test_ns = test_ns['target']
	x_vali_ns = vali_ns['Review']
	y_vali_ns = vali_ns['target']

	save_path = "C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 2\\data\\"

	# 1. unigram
	clf, count_vect, tfidf = train_data(x_train, y_train, vect_unigram, 1)
	test_scores['test_unigram'] = evaluate(x_test, y_test, clf, count_vect, tfidf)

	with open(save_path + "mnb_uni.pkl", 'wb') as file:
		pickle.dump(clf, file)

	best_alpha = alpha_tuning(x_train,y_train,x_vali,y_vali,vect_unigram,alpha_list)
	print("The selected alpha:{}\n".format(best_alpha))
	clf, count_vect, tfidf = train_data(x_train, y_train, vect_unigram, best_alpha)
	test_scores_optimized['test_unigram'] = evaluate(x_test, y_test, clf, count_vect, tfidf)

	# 2. bigram
	clf, count_vect, tfidf = train_data(x_train, y_train, vect_bigram, 1)
	test_scores['test_bigram'] = evaluate(x_test, y_test, clf, count_vect, tfidf)

	with open(save_path + "mnb_bi.pkl", 'wb') as file:
		pickle.dump(clf, file)

	best_alpha = alpha_tuning(x_train,y_train,x_vali,y_vali,vect_bigram,alpha_list)
	print("The selected alpha:{}\n".format(best_alpha))
	clf, count_vect, tfidf = train_data(x_train, y_train, vect_bigram, best_alpha)
	test_scores_optimized['test_bigram'] = evaluate(x_test, y_test, clf, count_vect, tfidf)

	# 3. unigram + bigram
	clf, count_vect, tfidf = train_data(x_train, y_train, vect_unibigram, 1)
	test_scores['test_unibigram'] = evaluate(x_test, y_test, clf, count_vect, tfidf)

	with open(save_path + "mnb_uni_bi.pkl", 'wb') as file:
		pickle.dump(clf, file)

	best_alpha = alpha_tuning(x_train,y_train,x_vali,y_vali,vect_unibigram,alpha_list)
	print("The selected alpha:{}\n".format(best_alpha))
	clf, count_vect, tfidf = train_data(x_train, y_train, vect_unibigram, best_alpha)
	test_scores_optimized['test_unibigram'] = evaluate(x_test, y_test, clf, count_vect, tfidf)

	# 4. unigram(no stopword)
	clf, count_vect, tfidf = train_data(x_train_ns, y_train_ns, vect_unigram, 1)
	test_scores['test_ns_uigram'] = evaluate(x_test_ns, y_test_ns, clf, count_vect, tfidf)

	with open(save_path + "mnb_uni_ns.pkl", 'wb') as file:
		pickle.dump(clf, file)

	best_alpha = alpha_tuning(x_train,y_train,x_vali_ns,y_vali_ns,vect_unigram,alpha_list)
	print("The selected alpha:{}\n".format(best_alpha))
	clf, count_vect, tfidf = train_data(x_train_ns, y_train_ns, vect_unigram, best_alpha)
	test_scores_optimized['test_ns_unigram'] = evaluate(x_test_ns, y_test_ns, clf, count_vect, tfidf)

	# 5. bigram(no stopword)
	clf, count_vect, tfidf = train_data(x_train_ns, y_train_ns, vect_bigram, 1)
	test_scores['test_ns_bigram'] = evaluate(x_test_ns, y_test_ns, clf, count_vect, tfidf)

	with open(save_path + "mnb_bi_ns.pkl", 'wb') as file:
		pickle.dump(clf, file)

	best_alpha = alpha_tuning(x_train,y_train,x_vali_ns,y_vali_ns,vect_bigram,alpha_list)
	print("The selected alpha:{}\n".format(best_alpha))
	clf, count_vect, tfidf = train_data(x_train_ns, y_train_ns, vect_bigram, best_alpha)
	test_scores_optimized['test_ns_bigram'] = evaluate(x_test_ns, y_test_ns, clf, count_vect, tfidf)

	# 6. unigram + bigram(no stopword)
	clf, count_vect, tfidf = train_data(x_train_ns, y_train_ns, vect_unibigram, 1)
	test_scores['test_ns_unibigram'] = evaluate(x_test_ns, y_test_ns, clf, count_vect, tfidf)

	with open(save_path + "mnb_uni_bi_ns.pkl", 'wb') as file:
		pickle.dump(clf, file)

	best_alpha = alpha_tuning(x_train,y_train,x_vali_ns,y_vali_ns,vect_unibigram,alpha_list)
	print("The selected alpha:{}\n".format(best_alpha))
	clf, count_vect, tfidf = train_data(x_train_ns, y_train_ns, vect_unibigram, best_alpha)
	test_scores_optimized['test_ns_unibigram'] = evaluate(x_test_ns, y_test_ns, clf, count_vect, tfidf)

	return test_scores, test_scores_optimized

if __name__ == "__main__":
	pprint(main(sys.argv[1]))