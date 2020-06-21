from gensim.models import Word2Vec
import sys

w2v_model = Word2Vec.load("C:\\Users\\andre\\msci-text-analytics-s20\\Assignment 3\\data\\w2v.model")

def main(file_path):
	with open (file_path) as f:
		words = f.read().lower().splitlines()
	for w in words:
		similar = w2v_model.similar_by_word(w, 20)
		print(w+":",similar)

if __name__ == "__main__":
	main(sys.argv[1])
