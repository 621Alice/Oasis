from Sentiment_Classifier.preprocessing.preprocessing_data_4labels import *
from functions import *
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import itertools, pickle

with open(p+'/Sentiment_Classifier/tokenizer/tokenizer_4labels.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes_4= ["happy", "sad", "hate","anger"]

model_test = load_model(p+'/model/checkpoint-4labels-0.618.h5')#supervised_model_v3_0.74_AfterDataCleaning

val_labels_4= np.argmax(val_labels_4, axis=1)
val_pred_4= model_test.predict(val_features_4)
val_pred_class_4= np.argmax(val_pred_4,axis=1)
print(classification_report(val_labels_4, val_pred_class_4, target_names=classes_4))

text = [ "@dannycastillo We want to trade with someone who has Houston tickets, but no one will.",
          "wants to hang out with friends SOON!",
        "Layin n bed with a headache  ughhhh...waitin on your call....",
        "It is so annoying when she starts typing on her computer in the middle of the night",
        "fuckin'm transtelecom",
        "@PandaMayhem noooooooooooo i just look at a lot of pictures   lol lol",
        "Can't sleep. Sucks. The one day i have to sleep in and i have to get up and go shopping with mom. Ugh."
       ]
#neutral, happy, sad,hate anger,happy, sad
sequences = tokenizer.texts_to_sequences(text)
padded_seq = pad_sequences(sequences, padding='post', maxlen=max_seq_len)
label_pred =model_test .predict(padded_seq)
for n, prediction in enumerate(label_pred):
    pred = label_pred.argmax(axis=-1)[n]
    print(text[n],"\nPrediction:",classes_4[pred],"\n")
