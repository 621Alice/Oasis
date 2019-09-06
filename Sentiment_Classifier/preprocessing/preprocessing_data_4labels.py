from functions import *
from sklearn.model_selection import train_test_split
token_num_words=40000

#read CSV file
texts=[]
sentiments=[]
with open(p+"/Data/data.csv", encoding='utf-8') as file:
    data=csv.reader(file, delimiter=",")
    for row in data:
          texts.append(data_cleaning(row[0]))
          sentiments.append(row[1])
print("finish reading data.csv")
i=0
for y in sentiments:
    if(y != '0'):
        features.append(texts[i])
        if(y== '1'):
            labels.append(0)
        elif (y == '2'):
            labels.append(1)
        elif (y == '3'):
            labels.append(2)
        elif (y == '4'):
            labels.append(3)
        else:
            print(y)
    i = i+1

features,labels,word_index = text2seq(features,labels,p+'/Sentiment_Classifier/tokenizer/tokenizer_4labels.pickle')
features,labels=shuffle(features,labels)


#split the training set and validation set
# num_validation=int(features.shape[0] * validation_ratio)
# print("num_validation:",num_validation)
# train_features_4=features[:-num_validation]
# train_labels_4=labels[:-num_validation]
# val_features_4=features[-num_validation:]
# val_labels_4=labels[-num_validation:]
# print("tarining labels:", train_labels_4.shape, train_labels_4.sum(axis=0),"\nvalidation labels:", val_labels_4.shape, val_labels_4.sum(axis=0))


train_features_4, val_features_4, train_labels_4, val_labels_4 = train_test_split(features, labels, test_size = 0.33, random_state = 0)
print(train_features_4.shape," ", val_features_4.shape, " ", train_labels_4.shape, " ", val_labels_4.shape)
