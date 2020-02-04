from tensorflow import keras

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

# Reverse the dict from [word][interger value] to [interger value][word] so that we can get the word 
# easily later
intToWord = dict([(value, key) for (key, value) in word_index.items()])

def return_processed_data_and_labels(maxLen):
    pre_train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=maxLen)
    pre_test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=maxLen)
    return [pre_train_data, train_labels, pre_test_data, test_labels]