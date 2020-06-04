import nltk
from nltk.corpus import stopwords

import random
import numpy as np

# Using stop words from nltk
stop_words = set(stopwords.words('english'))

# Remove the following special characters
# Note: Use a quadruple backslash (\\\\) to remove the backslash
special_characters = '[,.-?!"#$%&(*)+/:;<=>@\[\]\\\\^`{|}~\t\n]+' 

def tokenize(text, characters, lower = 1, with_stop_words = 1):
    # Remove the special characters
    # set punctuations as separate tokens
    for c in characters:
        text = text.replace(c," "+c+" ")
    # Split into tokens
    if lower == 0:
        tokens = text.split()
    elif lower == 1:
        # Normalize to lower case
        tokens = text.lower().split()   
    else:
        print("lower = 0: do not normalize to lower case \n")
        print("lower = 1: normalize to lower case")
    if with_stop_words == 1:
        return tokens
    elif with_stop_words == 0:
        tokens_without_sw = [w for w in tokens if not w in stop_words] 
        return tokens_without_sw
    else:
        print("with_stop_words = 0: without stopwords \n")
        print("with_stop_words = 1: with stopwords")


# split data into training, validation, and test sets
def split_data(data, training_size, test_size):
    if training_size < 0 or test_size < 0:
        print("Please input non-negtive values")
    elif training_size + test_size > 1:
        print("Please make sure that training_size + test_size <= 1")
    else:
        # shuffle the data into random order
        random.shuffle(data)
        data_size = len(data)
        training = data[:int(data_size*training_size)]
        test = data[int(data_size*training_size):int(data_size*(training_size+test_size))]
        validation = data[int(data_size*(training_size+test_size)):]
        return training, test, validation


def main():
    save_path = "C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 1\\"
    with open ("C:\\Users\\andre\\msci-text-analytics-s20\\pos.txt") as f:
        pos_lines = f.readlines()
    with open ("C:\\Users\\andre\\msci-text-analytics-s20\\neg.txt") as f:
        neg_lines = f.readlines()

    data = []
    data_no_stopword = []

    for line in pos_lines:
        tokens = tokenize(line, special_characters, lower = 1, with_stop_words = 1)
        tokens_no_stopword = tokenize(line, special_characters, lower = 1, with_stop_words = 0)
        data.append(tokens)
        data_no_stopword.append(tokens_no_stopword)

    pos_train, pos_test, pos_validation = split_data(data, 0.8, 0.1)
    pos_train_no_stopword, pos_test_no_stopword, pos_validation_no_stopword = split_data(data_no_stopword, 0.8, 0.1)

    np.savetxt(save_path + "pos_train.csv", pos_train, delimiter=",", fmt='%s')
    np.savetxt(save_path + "pos_test.csv", pos_test, delimiter=",", fmt='%s')
    np.savetxt(save_path + "pos_validation.csv", pos_validation, delimiter=",", fmt='%s')
    np.savetxt(save_path + "pos_train_no_stopword.csv", pos_train_no_stopword, delimiter=",", fmt='%s')
    np.savetxt(save_path + "pos_test_no_stopword.csv", pos_test_no_stopword, delimiter=",", fmt='%s')
    np.savetxt(save_path + "pos_validation_no_stopword.csv", pos_validation_no_stopword, delimiter=",", fmt='%s')

    data = []
    data_no_stopword = []

    for line in neg_lines:
        tokens = tokenize(line, special_characters, lower = 1, with_stop_words = 1)
        tokens_no_stopword = tokenize(line, special_characters, lower = 1, with_stop_words = 0)
        data.append(tokens)
        data_no_stopword.append(tokens_no_stopword)

    neg_train, neg_test, neg_validation = split_data(data, 0.8, 0.1)
    neg_train_no_stopword, neg_test_no_stopword, neg_validation_no_stopword = split_data(data_no_stopword, 0.8, 0.1)

    np.savetxt(save_path + "neg_train.csv", neg_train, delimiter=",", fmt='%s')
    np.savetxt(save_path + "neg_test.csv", neg_test, delimiter=",", fmt='%s')
    np.savetxt(save_path + "neg_validation.csv", neg_validation, delimiter=",", fmt='%s')
    np.savetxt(save_path + "neg_train_no_stopword.csv", neg_train_no_stopword, delimiter=",", fmt='%s')
    np.savetxt(save_path + "neg_test_no_stopword.csv", neg_test_no_stopword, delimiter=",", fmt='%s')
    np.savetxt(save_path + "neg_validation_no_stopword.csv", neg_validation_no_stopword, delimiter=",", fmt='%s')


if __name__ == "__main__":
    main()