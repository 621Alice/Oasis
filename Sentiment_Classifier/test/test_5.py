from Sentiment_Classifier.preprocessing.preprocessing_data_5labels import *
from functions import *
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

import itertools, pickle

with open(p+'/Sentiment_Classifier/tokenizer/tokenizer_5labels.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes_5= ["neutral", "happy", "sad", "hate","anger"]

#model_test = load_model(p+'/model/checkpoint-5labels-0.923.h5')#supervised_model_v2_0.64

model_test = load_model(p+'/model/checkpoint-5labels-0.935.h5')#supervised_model_v3_0.64_AfterDataCleaning
# model_test = load_model(p+'/model/checkpoint-5labels-0.928.h5')#supervised_model_v3_0.64_AfterDataCleaning_new

# model_test = load_model(p+'/model/checkpoint-5labels-0.929.h5')#supervised_model_v3_0.64_AfterDataCleaning_new_2



# model_test = load_model(p+'/model/checkpoint-seqCNNLSTM5labels-0.946.h5')#seqCNNLSTM
# model_test = load_model(p+'/model/checkpoint-seqCNN5labels-0.957.h5')#seqCNN
# model_test = load_model(p+'/model/checkpoint-seqLSTM5labels-0.940.h5')#seqLSTM

#model_test = load_model(p+'/model/checkpoint-5labels-0.938.h5')#GPU
#model_test = load_model(p+'/model/train_5&12_v1/checkpoint-5labels-0.925.h5')#supervised_model_v1
val_labels_5= np.argmax(val_labels_5, axis=1)
val_pred_5= model_test.predict(val_features_5)
val_pred_class_5= np.argmax(val_pred_5,axis=1)
print('Test accuracy is {}'.format(accuracy_score(val_labels_5, val_pred_class_5)))
print(classification_report(val_labels_5, val_pred_class_5, target_names=classes_5,digits=5))

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
    print(text[n],"\nPrediction:",classes_5[pred],"\n")
