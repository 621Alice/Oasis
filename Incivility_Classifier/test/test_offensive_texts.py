from Incivility_Classifier.preprocessing.preprocessing_offensive_texts import *
from functions import *
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np



with open(p+'/Incivility_Classifier/tokenizer/tokenizer_offensive.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes= ["non-offensive", "offensive"]
# model_test = load_model(p+'/model/checkpoint-toxic-texts-0.058.h5')
model_test = load_model(p+'/model/checkpoint-toxic-texts-0.052.h5') #new%
val_labels= np.argmax(y_test, axis=1)
val_pred= model_test.predict(X_test)
val_pred_class= np.argmax(val_pred,axis=1)
print(classification_report(val_labels, val_pred_class, target_names=classes, digits=5))
print('Test accuracy is {}'.format(accuracy_score(val_labels, val_pred_class)))


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
label_pred = model_test .predict(padded_seq)
for n, prediction in enumerate(label_pred):
    pred = label_pred.argmax(axis=-1)[n]
    print(text[n],"\nPrediction:",classes[pred],"\n")