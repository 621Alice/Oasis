from Sentiment_Classifier.preprocessing.preprocessing_data_7labels import *
from functions import *
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

import itertools, pickle

with open(p+'/Sentiment_Classifier/tokenizer/tokenizer_7labels.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes_7= ["worry","love","neutral","happy", "sad", "hate","anger"]

model_test = load_model(p+'/model/checkpoint-7labels-1.164.h5')#supervised_model_v3_0.56_AfterDataCleaning
# model_test = load_model(p+'/model/checkpoint-seqCNNLSTM7labels-1.210.h5')#seqCNNLSTM
# model_test = load_model(p+'/model/checkpoint-seqCNN7labels-1.235.h5')#seqCNN
# model_test = load_model(p+'/model/checkpoint-seqLSTM7labels-1.205.h5')#seqLSTM

val_labels_7= np.argmax(val_labels_7, axis=1)
val_pred_7= model_test.predict(val_features_7)
val_pred_class_7= np.argmax(val_pred_7,axis=1)
print('Test accuracy is {}'.format(accuracy_score(val_labels_7, val_pred_class_7)))
print(classification_report(val_labels_7, val_pred_class_7, target_names=classes_7))

text = [ "@dannycastillo We want to trade with someone who has Houston tickets, but no one will.",
          "top of the morning oranges! today be inspired, be creative and continue to belie...... ",
        "Layin n bed with a headache  ughhhh...waitin on your call....",
        "It is so annoying when she starts typing on her computer in the middle of the night",
        "fuckin'm transtelecom",
        " Why is playing 2 hands at once on the piano SO hard!",
        "Can't sleep. Sucks. The one day i have to sleep in and i have to get up and go shopping with mom. Ugh."
       ]

sequences = tokenizer.texts_to_sequences(text)
padded_seq = pad_sequences(sequences, padding='post', maxlen=max_seq_len)
label_pred =model_test .predict(padded_seq)
for n, prediction in enumerate(label_pred):
    pred = label_pred.argmax(axis=-1)[n]
    print(text[n],"\nPrediction:",classes_7[pred],"\n")
