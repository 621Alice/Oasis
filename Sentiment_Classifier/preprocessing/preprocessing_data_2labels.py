from functions import *
from sklearn.model_selection import train_test_split
#read CSV file
sentiments=[]
contents=[]
with open(p+'/Data/text_emotion.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        contents.append(data_cleaning(row[3]))
        sentiments.append(row[1])
contents=contents[1:]
sentiments=sentiments[1:]
i=0
for y in sentiments:
    if( y=="worry"):
        features.append(contents[i])
        labels.append(0)
    elif( y=="love"):
        features.append(contents[i])
        labels.append(1)
   
   
    i=i+1
print("finish reading text_emotion.csv")
#print(features[0])
#print(labels[0])



features,labels,word_index = text2seq(features,labels,p+'/Sentiment_Classifier/tokenizer/tokenizer_2labels.pickle')
features,labels=shuffle(features,labels)

#split the training set and validation set
# num_validation=int(features.shape[0] * validation_ratio)
# print("num_validation:",num_validation)
# train_features_2=features[:-num_validation]
# train_labels_2=labels[:-num_validation]
# val_features_2=features[-num_validation:]
# val_labels_2=labels[-num_validation:]
# print("tarining labels:", train_labels_2.shape, train_labels_2.sum(axis=0),"\nvalidation labels:", val_labels_2.shape, val_labels_2.sum(axis=0))

train_features_2, val_features_2, train_labels_2, val_labels_2 = train_test_split(features, labels, test_size = 0.33, random_state = 0)
print(train_features_2.shape," ", val_features_2.shape, " ", train_labels_2.shape, " ", val_labels_2.shape)

# train_features_2_1=features[:-num_validation]
# train_labels_2_1=labels[:-num_validation]
# val_features_2_1=features[-num_validation:]
# val_labels_2_1=labels[-num_validation:]
# print("validation labels 1 for test:", val_labels_2_1.shape, val_labels_2_1.sum(axis=0))
#
# train_features_2_2=features[num_validation:]
# train_labels_2_2=labels[num_validation:]
# val_features_2_2=features[:num_validation]
# val_labels_2_2=labels[:num_validation]
# print("validation labels 2 for test:", val_labels_2_2.shape, val_labels_2_2.sum(axis=0))
#
# train_features_2_3=features[:num_validation]
# train_labels_2_3=labels[:num_validation]
# for item in features[-num_validation:]:
#     np.insert(train_features_2_3,0,item)
# for item in labels[-num_validation:]:
#     np.insert(train_labels_2_3,0,item)
# val_features_2_3=features[num_validation:num_validation*2]
# val_labels_2_3=labels[num_validation:num_validation*2]
# print("validation labels 3 for test:", val_labels_2_3.shape, val_labels_2_3.sum(axis=0))


