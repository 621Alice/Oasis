from functions import *
from sklearn.model_selection import train_test_split
token_num_words=40000

#read CSV file

sentiments = []
contents = []
with open(p+'/Data/text_emotion.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        contents.append(data_cleaning(row[3]))
        sentiments.append(row[1])
contents = contents[1:]
sentiments = sentiments[1:]



i = 0
for y in sentiments:
    if (y == "worry"):
        features.append(contents[i])
        labels.append(0)
    elif (y == "love"):
        features.append(contents[i])
        labels.append(1)
    elif (y == "neutral"):
        features.append(contents[i])
        labels.append(2)

    i = i + 1
print("finish reading text_emotion.csv")
texts=[]
sentiments=[]
with open(p+"/Data/data.csv", encoding='utf-8') as file:
    data=csv.reader(file, delimiter=",")
    for row in data:
          texts.append(data_cleaning(row[0]))
          sentiments.append(row[1])

i=0
for y in sentiments:
    if(y != '0'):
        features.append(texts[i])
        if(y== '1'):
            labels.append(3)
        elif (y == '2'):
            labels.append(4)
        elif (y == '3'):
            labels.append(5)
        elif (y == '4'):
            labels.append(6)
        else:
            print(y)
    i = i+1
print("finish reading data.csv")


#for training with classifiers
# features,labels=shuffle_original(features,labels)
# classes_7= [0,1,2,3, 4, 5,6]
# # classes=["worry","love","neutral","happy", "sad", "hate","anger"]
# train_text=dict()
# train_label=dict()
# val_text=dict()
# val_label=dict()
# train_size=int(len(features)*validation_ratio)
# # print("training set:")
# for category in classes_7:
#     i = 0
#     for label in labels[:-train_size]:
#         # print(label," ", category)
#         if (label == category):
#             # print("yes")
#             if label in train_text.keys():
#                 # print(train_text)
#                 # print(train_label)
#                 train_text[label].append(features[i])
#                 train_label[label].append(labels[i])
#             else:
#
#                 train_text[label]= [features[i]]
#                 train_label[label]= [labels[i]]
#         i = i + 1
# # print(train_text)
# # print(train_label)
# # print("validation set:")
# val_features=features[-train_size:]
# val_sentimens=labels[-train_size:]
# for category in classes_7:
#
#
#
#     for j,label in enumerate(val_sentimens):
#
#         if (label == category):
#             # print(label, " ", category," ",val_labels[j])
#             # print(classes[labels[j]], " ", val_features[j])
#             if label in val_text.keys():
#
#                 val_text[label].append(val_features[j])
#                 val_label[label].append(val_sentimens[j])
#                 # print(classes[labels[j]], " ", features[j])
#             else:
#
#                 val_text[label] = [val_features[j]]
#                 val_label[label] = [val_sentimens[j]]
#                 # print(classes[labels[j]]," ",features[j])
# # print(val_text)
# # print(val_label)
# # for category in classes_7:
# #     train_text[category], label_t, word_index = text2seq(train_text[category], train_label[category], p + '/tokenizer/tokenizer_7labels.pickle')
# #     train_text[category], label_t = shuffle(train_text[category], label_t)
# #     # print("training labels for classifiers:", train_text[category].shape, label_t.sum(axis=0))
# #     val_text[category], label_v, word_index = text2seq(val_text[category], val_label[category], p + '/tokenizer/tokenizer_7labels.pickle')
# #     val_text[category], label_v = shuffle(val_text[category], label_v)
# #     # print("validation labels for classifiers:", val_text[category].shape, label_v.sum(axis=0))
#
#
#
# print("finish dividing texts into class groups")


#for training with CNN-LSTM
features,labels,word_index = text2seq(features,labels,p+'/Sentiment_Classifier/tokenizer/tokenizer_7labels.pickle')
features,labels=shuffle(features,labels)


#split the training set and validation set
# num_validation=int(features.shape[0] * validation_ratio)
# print("num_validation:",num_validation)
# train_features_7=features[:-num_validation]
# train_labels_7=labels[:-num_validation]
# val_features_7=features[-num_validation:]
# val_labels_7=labels[-num_validation:]
# print("tarining labels:", train_labels_7.shape, train_labels_7.sum(axis=0),"\nvalidation labels:", val_labels_7.shape, val_labels_7.sum(axis=0))
#

train_features_7, val_features_7, train_labels_7, val_labels_7 = train_test_split(features, labels, test_size = 0.33, random_state = 0)
print(train_features_7.shape," ", val_features_7.shape, " ", train_labels_7.shape, " ", val_labels_7.shape)
