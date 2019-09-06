from Sentiment_Classifier.preprocessing.preprocessing_data_8labels import *
#from preprocessing_data_8labels_equal_distribution import *
from functions import *
from keras.models import load_model
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import itertools, pickle

with open(p+'/Sentiment_Classifier/tokenizer/tokenizer_8labels.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes_8= ["neutral", "sadness","hate","worry","fun","love", "happiness","relief"]
# val_labels_8_1_support=val_labels_8_1.sum(axis=0)
# val_labels_8_2_support=val_labels_8_2.sum(axis=0)
# val_labels_8_3_support=val_labels_8_3.sum(axis=0)

model_test = load_model(p+'/model/checkpoint-8labels-1.603.h5')#supervised_model_v2
#model_test = load_model(p+'/model/checkpoint-8labels-2.090.h5')#supervised_model_v2_equal_distribution

val_labels_8= np.argmax(val_labels_8, axis=1)
val_pred_8= model_test.predict(val_features_8)
val_pred_class_8= np.argmax(val_pred_8,axis=1)
print(classification_report(val_labels_8, val_pred_class_8, target_names=classes_8))

# val_labels_8_1= np.argmax(val_labels_8_1, axis=1)
# val_pred_8_1= model_test.predict(val_features_8_1)
# val_pred_class_8_1= np.argmax(val_pred_8_1,axis=1)
# #print(classification_report(val_labels_8_1, val_pred_class_8_1, target_names=classes_8))
# pre_1=precision_score(val_labels_8_1, val_pred_class_8_1,average=None)
# rec_1=recall_score(val_labels_8_1, val_pred_class_8_1,average=None)
# f1_1=f1_score(val_labels_8_1, val_pred_class_8_1,average=None)
# #print(np.asarray(f1_1)[0])
# #print(val_labels_8_1_support[0])
# print("validation set 1 \nprecision for 8 labels:",pre_1,"\narecall for 8 labels:", rec_1, "\nF1 score for 8 labels:", f1_1)
#
# val_labels_8_2= np.argmax(val_labels_8_2, axis=1)
# val_pred_8_2= model_test.predict(val_features_8_2)
# val_pred_class_8_2= np.argmax(val_pred_8_2,axis=1)
# #print(classification_report(val_labels_8_2, val_pred_class_8_2, target_names=classes_8))
# pre_2=precision_score(val_labels_8_2, val_pred_class_8_2,average=None)
# rec_2=recall_score(val_labels_8_2, val_pred_class_8_2,average=None)
# f1_2=f1_score(val_labels_8_2, val_pred_class_8_2,average=None)
# print("validation set 2\nprecision for 8 labels:",pre_2,"\nrecall for 8 labels:", rec_2, "\nF1 score for 8 labels:", f1_2)
#
# val_labels_8_3= np.argmax(val_labels_8_3, axis=1)
# val_pred_8_3= model_test.predict(val_features_8_3)
# val_pred_class_8_3= np.argmax(val_pred_8_3,axis=1)
# #print(classification_report(val_labels_8_3, val_pred_class_8_3, target_names=classes_8))
# pre_3=precision_score(val_labels_8_3, val_pred_class_8_3,average=None)
# rec_3=recall_score(val_labels_8_3, val_pred_class_8_3,average=None)
# f1_3=f1_score(val_labels_8_3, val_pred_class_8_3,average=None)
# print("validation set 3 \nprecision for 8 labels:",pre_3,"\nrecall for 8 labels:", rec_3, "\nF1 score for 8 labels:", f1_3)
#
# pre=[]
# rec=[]
# f1=[]
# avg_f1_1=0
# avg_f1_2=0
# avg_f1_3=0
# print("\t\tprecision :","\t\trecall:","\t\t\tF1-score:")
# for i in range(len(classes_8)):
#
#     pre.append((pre_1[i]+pre_2[i]+pre_3[i])/3)
#     rec.append((rec_1[i]+rec_2[i]+rec_3[i])/3)
#     f1.append((f1_1[i]+f1_2[i]+f1_3[i])/3)
#     if(i==6):
#         print(classes_8[i],"\t{0:.3f}".format(pre[i]),"\t\t\t{0:.3f}".format(rec[i]),"\t\t\t{0:.3f}".format(f1[i]))
#     else:
#         print(classes_8[i],"\t\t{0:.3f}".format(pre[i]),"\t\t\t{0:.3f}".format(rec[i]),"\t\t\t{0:.3f}".format(f1[i]))
# for i in range(8):
#     avg_f1_1=avg_f1_1+(np.asarray(f1_1)[i]*val_labels_8_1_support[i]/num_validation)
#     avg_f1_2=avg_f1_2+(np.asarray(f1_2)[i]*val_labels_8_2_support[i]/num_validation)
#     avg_f1_3=avg_f1_2+(np.asarray(f1_3)[i]*val_labels_8_3_support[i]/num_validation)
# #print("F1-score for validation set 1: {0:.3f}".format(float(avg_f1_1)))
# #print("F1-score for validation set 2: {0:.3f}".format(float( avg_f1_2)))
# #print("F1-score for validation set 3: {0:.3f}".format(float(avg_f1_3)))
#
# print("F1-score in average: {0:.3f}".format( (float(avg_f1_1)+float(avg_f1_2)+float(avg_f1_3))/3))
#
#
#
#
#
#
# i=0
# for marker in ['o', '.', 's', 'x', '+', 'v', '^', '<']:
#     plt.plot(pre[i], rec[i], marker,
#              label="label='{0}'".format(classes_8[i]))
#     i=i+1
# plt.legend(numpoints=1)
# plt.xlim([0.00, 1.10])
# plt.ylim([0.00, 1.00])
# plt.xlabel('precision')
# plt.ylabel('recall')
# plt.title(' Precision-Recall plot to 8-label sentiment classfication')
# plt.show()
#
