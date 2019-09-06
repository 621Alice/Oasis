from Sentiment_Classifier.preprocessing.preprocessing_data_2labels import *

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
conv_1=MaxPooling1D(2)(conv_1)
conv_1 = Dropout(0.1)(conv_1)
merge_1=Concatenate(axis=1)([ conv_1,lstm_1])
lstm_2= Bidirectional(LSTM(6,dropout=0.1,recurrent_dropout=0.0,return_sequences=True))(merge_1)
conv_2 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(lstm_2)
conv_2=MaxPooling1D(2)(conv_2)
conv_2 = Dropout(0.1)(conv_2)
merge_2=Concatenate(axis=1)([ conv_1,lstm_1,conv_2,lstm_2])



conv_5 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences)
conv_5=MaxPooling1D(2)(conv_5)
conv_5 = Dropout(0.1)(conv_5)
lstm_5 = Bidirectional(LSTM(6,dropout=0.1,recurrent_dropout=0.0,return_sequences=True))(conv_5)
merge_5=Concatenate(axis=1)([lstm_5,conv_5])
conv_6 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(merge_5)
conv_6=MaxPooling1D(2)(conv_6)
conv_6 = Dropout(0.1)(conv_6)
lstm_6 = Bidirectional(LSTM(6,dropout=0.1,recurrent_dropout=0.0,return_sequences=True))(conv_6)
merge_6=Concatenate(axis=1)([lstm_5,conv_5,lstm_6,conv_6])


lstm_9 = Bidirectional(LSTM(6,dropout=0.05,recurrent_dropout=0.0,return_sequences=True))(embedded_sequences)
lstm_10 = Bidirectional(LSTM(6,dropout=0.05,recurrent_dropout=0.0,return_sequences=True))(lstm_9)

conv_9 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences)
conv_9 = MaxPooling1D(2)(conv_9)
conv_9 = Dropout(0.05)(conv_9)
conv_10 = Conv1D(filters=12,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(conv_9)
conv_10 = MaxPooling1D(2)(conv_10)
conv_10 = Dropout(0.05)(conv_10)



merge=Concatenate(axis=1)([merge_2,merge_6,lstm_10,conv_10])
pool= MaxPooling1D(4)(merge)
drop= Dropout(0.4)(pool)
flat = Flatten()(drop)
dense = Dense(24, activation='relu')(flat)
preds = Dense(2, activation='softmax')(dense)

model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1.0, epsilon=None, decay=0.000)


model_checkpoints = callbacks.ModelCheckpoint(p+"/model/checkpoint-2labels-{val_loss:.3f}.h5", verbose=0,period=0,monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto')
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['acc'])

print("Training Progress:")
model_log = model.fit(train_features_2,train_labels_2, validation_data=(val_features_2,val_labels_2),
          epochs=30, batch_size=200,
         callbacks=[model_checkpoints])
