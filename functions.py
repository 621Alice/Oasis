#from __future__ import print_function

import math
import collections
import io
import math

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
#from IPython import display
from sklearn import metrics
import re, sys, os, csv, keras, pickle

from keras import regularizers, initializers, optimizers, callbacks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding,  Dropout, LSTM, GRU, Bidirectional
from keras.layers import Concatenate
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

p="/Users/liuleyu/Oasis"


pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

tf.logging.set_verbosity(tf.logging.ERROR)

token_num_words=40000#num_words for tokenizer
max_seq_len=40#maximum length for the padded sequence(for each text)
features,labels=[],[]
validation_ratio=0.33#the ratio of the validation set in the original dataset



#reshape the tweets to word-to-index sequences with post-sequence padding to max_seq_len
def text2seq(texts,labels,tokenizer_file):
    #remove punctuation(except ') and turn texts into integers(index of tokens in the dictionary) with max_num_words ranked by the frequency
    tokenizer = Tokenizer(num_words=token_num_words)
    #list of texts to train on
    tokenizer.fit_on_texts(texts)
    #serialize
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #dictionary mapping words (str) to their index (int). 
    word_index=tokenizer.word_index
    #turn list of texts into sequences
    sequences = tokenizer.texts_to_sequences(texts)
    padded_text=pad_sequences(sequences, padding='post', maxlen=max_seq_len)
    #convert intergers to binary class matrix(one-hot encoding)
    labels = to_categorical(np.asarray(labels))
    #print(padded_text[0])
    #print(labels[0])
    return padded_text, labels, word_index


#shuffle the indices for 2d-array features and labels
def shuffle(texts,labels):
  
   indices= np.arange(texts.shape[0])
   np.random.shuffle(indices)
   np.random.shuffle(indices)
   texts=texts[indices]
   labels=labels[indices]
   print("texts:",texts.shape)
   print("labels:",labels.shape)
   return texts,labels


#shuffle the indices for original list
def shuffle_original(texts,labels):
   texts_shuffle=[]
   labels_shuffle=[]
   indices= np.arange(len(texts))
   np.random.shuffle(indices)
   np.random.shuffle(indices)
   for i in indices:
       texts_shuffle.append(texts[i])
       labels_shuffle.append(labels[i])

   print("texts:",len(texts_shuffle))
   print("labels:",len(labels_shuffle))
   return texts_shuffle,labels_shuffle

#build embedding layer
def build_embeddings(embedding_dim, word_index):
    embedding_indices={}


    f=open(p+'/Data/glove.twitter.27B.'+str(embedding_dim)+'d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        #creating embedding_indices{word:vector}
        embedding_indices[word] = np.asarray(values[1:], dtype='float32')
    f.close()
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_indices.get(word)
        if embedding_vector is not None:
            # map word to vector with the same i(the index of word in dictionary) and words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print("finished embeddings!")
    return embedding_matrix



#cleaning the dataset
def data_cleaning(words):
    words=str(words).lower()
    stopwords_en = set(stopwords.words("english"))

    cleaned_data_1=[word for word in words.split(' ')
                    if 'http' not in word
                    and not word.startswith('@')
                    and not word.startswith('http')
                    and not word.startswith('https')
                    and not word.startswith('#')
                    and word != 'rt']
    cleaned_data_2=' '.join([word for word in cleaned_data_1 if not word in stopwords_en])
    return cleaned_data_2











