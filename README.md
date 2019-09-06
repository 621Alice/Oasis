# Sentiment-Analysis-and-Incivility-Detection


#Programming Language
Python 3.7

#Library
Tensorflow
numpy
Keras
Sklearn
NLTK
Pandas
Matplotlib


#Configuration
CPU: 2.3GHz dual-core Intel Core i5
Storage: 256GB

#run
1. Please download the pre-trained word embedding : "glove.twitter.27B.200d.txt" via https://nlp.stanford.edu/projects/glove/ and put it under "Data" directory
2. Please change the file path p (line 32, function.py) to the current path of this project directory  before running any files

#Incivility Classifier
Under "Incivility_Classifier" directory:
train the classifier by running "train_offensive_texts.py"
test the classifier by running "test_offensive_texts.py"

#Sentiment Classifier
Under "Sentiment_Classifier" directory:
train the classifier by running files under "train" directory
test the classifier by running files under "test" directory

#Data
Under "Data" directory:
Two datasets used for training Sentiment Classifier: "data.csv" and "text_emotion.csv"
one dataset used for training Incivility Classifier: "twitter-hate-speech-classifier-DFE-a84500.csv"


#model
Under "model" directory:
trained models of the two classifiers
