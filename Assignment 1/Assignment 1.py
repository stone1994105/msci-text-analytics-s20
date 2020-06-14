import random

with open ("C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 1\\stopwords.txt") as f:
    stop_words = f.read()

# Remove the following special characters
# Note: Use a quadruple backslash (\\\\) to remove the backslash
special_characters = '[\'\",.-?!"#$%&(*)+/:;<=>@\[\]\\\\^`{|}~\t\n]+' 

def tokenize(text, characters, lower = 1, with_stop_words = 1):
    # Remove the special characters
    for c in characters:
        text = text.replace(c," ")
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
    random.seed(2020)
    save_path = "C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 1\\data\\"
    with open ("C:\\Users\\andre\\msci-text-analytics-s20\\pos.txt") as f:
        pos_lines = f.readlines()
    with open ("C:\\Users\\andre\\msci-text-analytics-s20\\neg.txt") as f:
        neg_lines = f.readlines()

    data = []
    data_ns = []

    for line in pos_lines:
        tokens = tokenize(line, special_characters, lower = 1, with_stop_words = 1)
        tokens_ns = tokenize(line, special_characters, lower = 1, with_stop_words = 0)
        data.append(tokens)
        data_ns.append(tokens_ns)

    pos_train, pos_test, pos_vali = split_data(data, 0.8, 0.1)
    pos_train_ns, pos_test_ns, pos_vali_ns = split_data(data_ns, 0.8, 0.1)

    data = []
    data_ns = []

    for line in neg_lines:
        tokens = tokenize(line, special_characters, lower = 1, with_stop_words = 1)
        tokens_ns = tokenize(line, special_characters, lower = 1, with_stop_words = 0)
        data.append(tokens)
        data_ns.append(tokens_ns)

    neg_train, neg_test, neg_vali = split_data(data, 0.8, 0.1)
    neg_train_ns, neg_test_ns, neg_vali_ns = split_data(data_ns, 0.8, 0.1)

    with open (save_path + "pos_train.csv", "w") as f :
        for sublist in pos_train:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "pos_test.csv", "w") as f :
        for sublist in pos_test:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "pos_vali.csv", "w") as f :
        for sublist in pos_vali:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "pos_train_ns.csv", "w") as f :
        for sublist in pos_train_ns:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "pos_test_ns.csv", "w") as f :
        for sublist in pos_test_ns:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "pos_vali_ns.csv", "w") as f :
        for sublist in pos_vali_ns:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')

    with open (save_path + "neg_train.csv", "w") as f :
        for sublist in neg_train:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "neg_test.csv", "w") as f :
        for sublist in neg_test:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "neg_vali.csv", "w") as f :
        for sublist in neg_vali:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "neg_train_ns.csv", "w") as f :
        for sublist in neg_train_ns:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "neg_test_ns.csv", "w") as f :
        for sublist in neg_test_ns:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')
    with open (save_path + "neg_vali_ns.csv", "w") as f :
        for sublist in neg_vali_ns:
            for item in sublist:
                f.write(item + ',')
            f.write('\n')

if __name__ == "__main__":
    main()