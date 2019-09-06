# Oasis

Oasis is an online analytic system for incivility detection and sentiment classification. This package provides the source codes of deep learning-based classification models in Oasis.
 
## Programming Language:

Python 3.7


## Library:

Tensorflow, numpy, Keras, Sklearn, NLTK, Pandas, Matplotlib


## Configuration:

CPU: 2.3GHz dual-core Intel Core i5

Storage: 256GB


## Run:

1. Please download the pre-trained word embedding : "glove.twitter.27B.200d.txt" via https://nlp.stanford.edu/projects/glove/ and put it under "Data" directory
2. Please change the file path p (line 32, function.py) to the current path of this project directory  before running any files


## Incivility Classifier:

Under "Incivility_Classifier" directory:
train the classifier by running "train_offensive_texts.py"
test the classifier by running "test_offensive_texts.py"


## Sentiment Classifier:

Under "Sentiment_Classifier" directory:
train the classifier by running files under "train" directory
test the classifier by running files under "test" directory


## Data:

Under "Data" directory:
Two datasets used for training Sentiment Classifier: "data.csv" and "text_emotion.csv"
one dataset used for training Incivility Classifier: "twitter-hate-speech-classifier-DFE-a84500.csv"


## model:

Under "model" directory:
trained models of the two classifiers

 

If you find that our codes are useful, please use the following BibTeX citations to cite our paper.

```
@inproceedings{lhxs2019oasis,

  title     = {Oasis: Online Analytic System for Incivility Detection and Sentiment },
  
  author    = {Liu, Leyu and Huang, Xin and Xu, Jianliang and Song, Yunya},
  
  booktitle = {IEEE International Conference on Data Mining},
  
  year      = {2019}
  
}
```
