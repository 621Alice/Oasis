from keras import Sequential
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


# from preprocessing.preprocessing_data_7labels import *
from Sentiment_Classifier.preprocessing.preprocessing_data_5labels import *

from sklearn import metrics
embedding_dim=200
embedding_matrix=build_embeddings(embedding_dim, word_index)
embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_seq_len,
                            trainable=False)

# sequence_input=Input(shape=(max_seq_len,), dtype='int32')
# embedded_sequences=embedding_layer(sequence_input)

#7-label
# def create_model(regularizer=regularizers.l2(0.0001),dropout=0.2):
    #CNNLSTM
    # model = Sequential()
    # model.add(embedding_layer)
    # model.add(Conv1D(kernel_size=2, filters=20, activation='relu', kernel_regularizer=regularizer))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Bidirectional(LSTM(32, input_shape=(10, 64))))

    #CNN
    # model = Sequential()
    # model.add(embedding_layer)
    # model.add(Conv1D(kernel_size=2, filters=20, activation='relu',kernel_regularizer=regularizer))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())

    #LSTM
    # model = Sequential()
    # model.add(embedding_layer)
    # model.add(Bidirectional(LSTM(32, input_shape=(10, 64))))
    # model.add(Dense(64,activation='relu',kernel_regularizer=regularizer))
    # model.add(Dropout(dropout))
    # model.add(Dense(7,activation='softmax'))
    # adadelta = optimizers.Adadelta(lr=1.0, epsilon=None, decay=0.000)
    # model.compile(loss='categorical_crossentropy',optimizer=adadelta, metrics=['accuracy'])

    # return model


#5-label
def create_model(regularizer=regularizers.l2(0.0001),dropout=0.2):
    #CNNLSTM
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(kernel_size=2, filters=20, activation='relu', kernel_regularizer=regularizer))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(32, input_shape=(10, 64))))

    #CNN
    # model = Sequential()
    # model.add(embedding_layer)
    # model.add(Conv1D(kernel_size=2, filters=20, activation='relu',kernel_regularizer=regularizer))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())

    #LSTM
    # model = Sequential()
    # model.add(embedding_layer)
    # model.add(Bidirectional(LSTM(32, input_shape=(10, 64))))

    model.add(Dense(64,activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(dropout))
    model.add(Dense(5,activation='softmax'))
    adadelta = optimizers.Adadelta(lr=1.0, epsilon=None, decay=0.000)
    model.compile(loss='categorical_crossentropy',optimizer=adadelta, metrics=['accuracy'])

    return model







#kearas sklearn classification wrapper

# clf = KerasClassifier(build_fn=create_model,verbose=0,epochs=30, batch_size=200)
#
# pipeline=Pipeline([
#
#     ('clf', clf)
# ])
# pipeline.fit(train_features_7,train_labels_7)
# scores = cross_val_score(pipeline, val_features_7,val_labels_7, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#Feed-forward NN
model=create_model()
#feed-forward CNN LSTM------checkpoint-seqCNNLSTM7labels
#feed-forward CNN------checkpoint-seqCNN7labels
#feed-forward LSTM------checkpoint-seqLSTM7labels

#7-label
# model_checkpoints = callbacks.ModelCheckpoint(p+"/model/checkpoint-seqLSTM7labels-{val_loss:.3f}.h5", verbose=0,period=0,monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto')
# model.summary()
# model_log = model.fit(train_features_7,train_labels_7, validation_data=(val_features_7,val_labels_7),
#           epochs=30, batch_size=200,
#          callbacks=[model_checkpoints])


#5-label
model_checkpoints = callbacks.ModelCheckpoint(p+"/model/checkpoint-seqCNNLSTM5labels-{val_loss:.3f}.h5", verbose=0,period=0,monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto')
model.summary()
model_log = model.fit(train_features_5,train_labels_5, validation_data=(val_features_5,val_labels_5),
          epochs=30, batch_size=200,
         callbacks=[model_checkpoints])
