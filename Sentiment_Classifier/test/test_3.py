from Sentiment_Classifier.preprocessing.preprocessing_data_3labels import*
from functions import *
from keras.models import load_model
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import itertools, pickle

with open(p+'/Sentiment_Classifier/tokenizer/tokenizer_3labels.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes_3= ["worry", "love","neutral"]
# val_labels_3_1_support=val_labels_3_1.sum(axis=0)
# val_labels_3_2_support=val_labels_3_2.sum(axis=0)
# val_labels_3_3_support=val_labels_3_3.sum(axis=0)

#model_test = load_model(p+'/model/checkpoint-3labels-0.772.h5')#supervised_model_v2_GPU
#model_test = load_model(p+'/model/train_v2/checkpoint-3labels-0.828.h5')
# model_test = load_model(p+'/model/checkpoint-3labels-0.830.h5')#GPU
#model_test = load_model(p+'/model/checkpoint-3labels-0.816.h5')#supervised_model_v3
model_test = load_model(p+'/model/checkpoint-3labels-0.823.h5')#supervised_model_v3_new

val_labels_3= np.argmax(val_labels_3, axis=1) 
val_pred_3= model_test.predict(val_features_3)
val_pred_class_3= np.argmax(val_pred_3,axis=1)
print('Test accuracy is {}'.format(accuracy_score(val_labels_3, val_pred_class_3)))
print(classification_report(val_labels_3, val_pred_class_3, target_names=classes_3, digits=10))


# val_labels_3_1= np.argmax(val_labels_3_1, axis=1)
# val_pred_3_1= model_test.predict(val_features_3_1)
# val_pred_class_3_1= np.argmax(val_pred_3_1,axis=1)
#
# pre_1=precision_score(val_labels_3_1, val_pred_class_3_1,average=None)
# rec_1=recall_score(val_labels_3_1, val_pred_class_3_1,average=None)
# f1_1=f1_score(val_labels_3_1, val_pred_class_3_1,average=None)
# print("validation set 1 \nprecision for 3 labels:",pre_1,"\nrecall for 3 labels:", rec_1, "\nF1 score for 3 labels:", f1_1)
#
# val_labels_3_2= np.argmax(val_labels_3_2, axis=1)
# val_pred_3_2= model_test.predict(val_features_3_2)
# val_pred_class_3_2= np.argmax(val_pred_3_2,axis=1)
#
# pre_2=precision_score(val_labels_3_2, val_pred_class_3_2,average=None)
# rec_2=recall_score(val_labels_3_2, val_pred_class_3_2,average=None)
# f1_2=f1_score(val_labels_3_2, val_pred_class_3_2,average=None)
# print("validation set 2\nprecision for 3 labels:",pre_2,"\nrecall for 3 labels:", rec_2, "\nF1 score for 3 labels:", f1_2)
#
# val_labels_3_3= np.argmax(val_labels_3_3, axis=1)
# val_pred_3_3= model_test.predict(val_features_3_3)
# val_pred_class_3_3= np.argmax(val_pred_3_3,axis=1)
#
# pre_3=precision_score(val_labels_3_3, val_pred_class_3_3,average=None)
# rec_3=recall_score(val_labels_3_3, val_pred_class_3_3,average=None)
# f1_3=f1_score(val_labels_3_3, val_pred_class_3_3,average=None)
# print("validation set 3 \nprecision for 3 labels:",pre_3,"\nrecall for 3 labels:", rec_3, "\nF1 score for 3 labels:", f1_3)
#
# pre=[]
# rec=[]
# f1=[]
# avg_f1_1=0
# avg_f1_2=0
# avg_f1_3=0
# print("\t\tprecision :","\t\trecall:","\t\t\tF1-score:")
# for i in range(len(classes_3)):
#
#     pre.append((pre_1[i]+pre_2[i]+pre_3[i])/3)
#     rec.append((rec_1[i]+rec_2[i]+rec_3[i])/3)
#     f1.append((f1_1[i]+f1_2[i]+f1_3[i])/3)
#     if(i==7 or i==8):
#         print(classes_3[i],"\t{0:.3f}".format(pre[i]),"\t\t\t{0:.3f}".format(rec[i]),"\t\t\t{0:.3f}".format(f1[i]))
#     else:
#         print(classes_3[i],"\t\t{0:.3f}".format(pre[i]),"\t\t\t{0:.3f}".format(rec[i]),"\t\t\t{0:.3f}".format(f1[i]))
#
# for i in range(3):
#     avg_f1_1=avg_f1_1+(f1_1[i]*val_labels_3_1_support[i]/num_validation)
#     avg_f1_2=avg_f1_2+(f1_2[i]*val_labels_3_2_support[i]/num_validation)
#     avg_f1_3=avg_f1_2+(f1_2[i]*val_labels_3_3_support[i]/num_validation)
# #print("F1-score for validation set 1: {0:.3f}".format(float(avg_f1_1)))
# #print("F1-score for validation set 2: {0:.3f}".format(float( avg_f1_2)))
# #print("F1-score for validation set 3: {0:.3f}".format(float(avg_f1_3)))
#
# print("F1-score in average: {0:.3f}".format( (float(avg_f1_1)+float(avg_f1_2)+float(avg_f1_3))/3))
#
# i=0
# for marker in ['+', 'v', '^']:
#     plt.plot(pre[i], rec[i], marker,
#              label="label='{0}'".format(classes_3[i]))
#     i=i+1
# plt.legend(numpoints=1)
# plt.xlim([0.00, 1.10])
# plt.ylim([0.00, 1.00])
# plt.xlabel('precision')
# plt.ylabel('recall')
# plt.title(' Precision-Recall plot to 3-label sentiment classfication')
# plt.show()
#
