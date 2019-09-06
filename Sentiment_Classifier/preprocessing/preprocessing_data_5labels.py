from functions import *
token_num_words=40000
from sklearn.model_selection import train_test_split

#read CSV file
original = []
with open(p+"/Data/data.csv", encoding='utf-8') as file:
    data=csv.reader(file, delimiter=",")
    for row in data:

          original.append(row[0])
          features.append(data_cleaning(row[0]))
          labels.append(row[1])
print("finish reading data.csv")

# for i in range(3):
#     print("Original text: "+original[i]+"\n")
#     print("Text after cleaning:"+features[i]+"\n"+"############"+"\n")


features,labels,word_index = text2seq(features,labels,p+'/Sentiment_Classifier/tokenizer/tokenizer_5labels.pickle')
features,labels=shuffle(features,labels)



#split the training set and validation set
# num_validation=int(features.shape[0] * validation_ratio)
# print("num_validation:",num_validation)
# train_features_5=features[:-num_validation]
# train_labels_5=labels[:-num_validation]
# val_features_5=features[-num_validation:]
# val_labels_5=labels[-num_validation:]
# print("tarining labels:", train_labels_5.shape, train_labels_5.sum(axis=0),"\nvalidation labels:", val_labels_5.shape, val_labels_5.sum(axis=0))

train_features_5, val_features_5, train_labels_5, val_labels_5 = train_test_split(features, labels, test_size = 0.33, random_state = 0)
print(train_features_5.shape," ", val_features_5.shape, " ", train_labels_5.shape, " ", val_labels_5.shape)
