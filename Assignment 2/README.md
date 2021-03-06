# MSCI-641 Assignment 2 Report
Name: Tianyu Shi<br />
Student ID: 20570373
| Stopwords Removed  | Text Features    | Accuracy (test set) |
| ------------------ | ---------------- | ------------------- |
| yes                | unigrams         |        80.13%       |
| yes                | bigrams          |        78.30%       |
| yes                | unigrams+bigrams |        80.44%       |
| no                 | unigrams         |        80.96%       |
| no                 | bigrams          |        82.23%       |
| no                 | unigrams+bigrams |        83.32%       |

a. _with stop words performed better_<br />
Sentiment analysis is actually sensitive to stopwords. Removing some stopwords that represent negative meanings will change the sentiment of the text. Consider the sentence "I don't like that product" as an example, where the meaning is opposite if we remove the stopwords "don't" and the sentence will be classified into the wrong category. The differences of the accuracy between with-stopwords condition and without-stopwords condition are 0.83%, 3.93%, and 2.88% for unigrams, bigrams, and unigrams+bigrams respectively. It seems like that stopwords become more influential to the accuracy when bigrams is introduced.



b. _unigrams+bigrams performed better_<br />
Since there exists meaningful phrases longer than one word, bigrams lets the model deain those features. However, using bigrams along tends to overfit the model so that the accuracy drops, and taking unigrams into consideration could reduce the bias of the model and make it more general. The bigrams only model performs worse than unigrams+bigrams model because the probability of the same bigrams repeating in different documents is lower. Combing bigrams with unigrams in the text features enables the model to capture longer-distance dependencies in comparison with unigrams only, which also reduces the probability of 
