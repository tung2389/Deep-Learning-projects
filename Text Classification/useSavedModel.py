import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
cwd = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 88000)

word_index = data.get_word_index() 

# Anwers for below preprocessing:
#  https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
#  https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

unknownCharacters = [",", ".", "(", ")", ":", '"', ";", "!"]

def encodeReview(review):
	encodedString = [1]
	for word in review:
		if word.lower() in word_index:
			encodedString.append(word_index[word.lower()])
		else:
			encodedString.append(2)
	return encodedString

model = keras.models.load_model(cwd + "/Text Classification/model.h5")

def preprocessReview(review):
	conciseString = review
	for char in unknownCharacters: 
		temp = conciseString.replace(char, "")
		conciseString = temp
	conciseString = conciseString.strip().split(" ")
	encodedString = encodeReview(conciseString)
	encodedString = keras.preprocessing.sequence.pad_sequences([encodedString], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
	return encodedString
	

def testModelWithRealData():
	with open(cwd + "/Text Classification/positive-review.txt", encoding="utf-8") as f:
		for line in f.readlines():
			processedReview = preprocessReview(line)
			predict = model.predict(processedReview)
			print("\n")
			print(line)
			print(processedReview)
			print(predict[0])

def runModel():
	review = input("Enter a movie review: ")
	processedReview = preprocessReview(review)
	predict = model.predict(processedReview)
	if predict < 0.4:
		print("This is a very negative review")
	elif 0.4 <= predict < 0.7:
		print("This is a both negative and positive review")
	else:
		print("This is a very positive review")

runModel()




