from Sentiment_Classifier.preprocessing.preprocessing_data_5labels import *

import pandas as pd
#build embedding layer
embedding_dim=200
embedding_matrix=build_embeddings(embedding_dim, word_index)
embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_seq_len,
                            trainable=False)

sequence_input=Input(shape=(max_seq_len,), dtype='int32')
embedded_sequences=embedding_layer(sequence_input)


lstm_1 = Bidirectional(LSTM(6,recurrent_dropout=0.0,return_sequences=True,dropout=0.15))(embedded_sequences)
conv_1 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(lstm_1)
conv_1 = Dropout(0.15)(conv_1)
merge_1=Concatenate(axis=1)([ conv_1,lstm_1])
lstm_2= Bidirectional(LSTM(6,dropout=0.15,recurrent_dropout=0.0,return_sequences=True))(merge_1)
conv_2 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(lstm_2)
conv_2 = Dropout(0.15)(conv_2)
merge_2=Concatenate(axis=1)([ conv_1,lstm_1,conv_2,lstm_2])
lstm_3= Bidirectional(LSTM(6,dropout=0.15,recurrent_dropout=0.0,return_sequences=True))(merge_2)
conv_3 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(lstm_3)
conv_3 = Dropout(0.15)(conv_3)
merge_3=Concatenate(axis=1)([ conv_1,lstm_1,conv_2,lstm_2,conv_3,lstm_3])
lstm_4= Bidirectional(LSTM(6,dropout=0.15,recurrent_dropout=0.0,return_sequences=True))(merge_3)
conv_4 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(lstm_4)
conv_4 = Dropout(0.15)(conv_3)
merge_4=Concatenate(axis=1)([ conv_1,lstm_1,conv_2,lstm_2,conv_3,lstm_3,conv_4,lstm_4])


conv_5 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences)
conv_5 = Dropout(0.15)(conv_5)
lstm_5 = Bidirectional(LSTM(6,dropout=0.15,recurrent_dropout=0.0,return_sequences=True))(conv_5)
merge_5=Concatenate(axis=1)([lstm_5,conv_5])
conv_6 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(merge_5)
conv_6 = Dropout(0.15)(conv_6)
lstm_6 = Bidirectional(LSTM(6,dropout=0.15,recurrent_dropout=0.0,return_sequences=True))(conv_6)
merge_6=Concatenate(axis=1)([lstm_5,conv_5,lstm_6,conv_6])
conv_7 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(merge_6)
conv_7 = Dropout(0.15)(conv_7)
lstm_7 = Bidirectional(LSTM(6,dropout=0.15,recurrent_dropout=0.0,return_sequences=True))(conv_7)
merge_7=Concatenate(axis=1)([lstm_5,conv_5,lstm_6,conv_6,lstm_7,conv_7])
conv_8 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(merge_7)
conv_8 = Dropout(0.15)(conv_8)
lstm_8 = Bidirectional(LSTM(6,dropout=0.15,recurrent_dropout=0.0,return_sequences=True))(conv_8)
merge_8=Concatenate(axis=1)([lstm_5,conv_5,lstm_6,conv_6,lstm_7,conv_7,lstm_8,conv_8])

lstm_9 = Bidirectional(LSTM(6,dropout=0.05,recurrent_dropout=0.0,return_sequences=True))(embedded_sequences)
lstm_10 = Bidirectional(LSTM(6,dropout=0.05,recurrent_dropout=0.0,return_sequences=True))(lstm_9)
lstm_11 = Bidirectional(LSTM(6,dropout=0.05,recurrent_dropout=0.0,return_sequences=True))(lstm_10)
lstm_12 = Bidirectional(LSTM(6,dropout=0.05,recurrent_dropout=0.0,return_sequences=True))(lstm_11)

conv_9 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences)
conv_9 = MaxPooling1D(2)(conv_9)
conv_9 = Dropout(0.05)(conv_9)
conv_10 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(conv_9)
conv_10 = MaxPooling1D(2)(conv_10)
conv_10 = Dropout(0.05)(conv_10)
conv_11 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(conv_10)
conv_11 = MaxPooling1D(2)(conv_11)
conv_11 = Dropout(0.05)(conv_11)
conv_12 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(conv_10)
conv_12 = MaxPooling1D(2)(conv_12)
conv_12 = Dropout(0.05)(conv_12)



merge=Concatenate(axis=1)([merge_4,merge_8,lstm_11,conv_11])
pool= MaxPooling1D(4)(merge)
drop= Dropout(0.4)(pool)
flat = Flatten()(drop)
dense = Dense(24, activation='relu')(flat)
preds = Dense(5, activation='softmax')(dense)

model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1.0, epsilon=None, decay=0.000)


model_checkpoints = callbacks.ModelCheckpoint(p+"/model/checkpoint-5labels-{val_loss:.3f}.h5", verbose=0,period=0,monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto')
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['acc'])

print("Training Progress:")
model_log = model.fit(train_features_5,train_labels_5, validation_data=(val_features_5,val_labels_5),
          epochs=30, batch_size=200,
         callbacks=[model_checkpoints])
