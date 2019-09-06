from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from Sentiment_Classifier.preprocessing.preprocessing_data_7labels import *

data= pd.read_csv(p+'/Data/data.csv', encoding='utf-8')
# print(data.head())
texts=[]

y=data.iloc[:,1].values
# print(y[0:5])
# print(data.iloc[:,0].values[0:5])
for row in data.iloc[:,0].values:
    # print(row)
    texts.append(data_cleaning(row))

cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(texts).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
print(X_train.shape," ", X_test.shape, " ", y_train.shape, " ", y_test.shape)

NB_pipeline = Pipeline([

                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])


NB_pipeline.fit(X_train, y_train)

prediction = NB_pipeline.predict(X_test)
print('NB Test accuracy is {}'.format(accuracy_score(y_test, prediction)))

LogReg_pipeline = Pipeline([

                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])
LogReg_pipeline.fit(X_train, y_train)

prediction = LogReg_pipeline.predict(X_test)
print('LogReg Test accuracy is {}'.format(accuracy_score(y_test, prediction)))

SVC_pipeline = Pipeline([

                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])

SVC_pipeline.fit(X_train, y_train)

prediction = SVC_pipeline.predict(X_test)
print('SVC Test accuracy is {}'.format(accuracy_score(y_test, prediction)))



