from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from functions import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

data= pd.read_csv(p+'/Data/twitter-hate-speech-classifier-DFE-a845520.csv', encoding='ISO-8859-1')
# print(data.head())
texts=[]
if_toxic=[]

for row in data.iloc[:,-1].values:
    # print(row)
    texts.append(data_cleaning(row))


for row in data.iloc[:,5].values:
    # print(row)
    if "not offensive" in row:
        if_toxic.append(0)

    else:
        if_toxic.append(1)
# print(if_toxic[0]," ",texts[0])

# print(if_toxic[20:30]," ",texts[20:30])

features,labels,word_index = text2seq(texts,if_toxic,p+'/Incivility_Classifier/tokenizer/tokenizer_offensive.pickle')

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.33, random_state = 0)
print(X_train.shape," ", X_test.shape, " ", y_train.shape, " ", y_test.shape)
