import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential

# load the dataset but only keep the top n words, zero the rest
topWords = 5000
(Xtrain, Ytrain), (Xtest, Ytest) = imdb.load_data(num_words=topWords)

# truncate and pad input sequences
maxReviewLength = 500
Xtrain = sequence.pad_sequences(Xtrain, maxlen=maxReviewLength)
Xtest = sequence.pad_sequences(Xtest, maxlen=maxReviewLength)

# create model
embeddingVectorLength = 32
model = Sequential()
