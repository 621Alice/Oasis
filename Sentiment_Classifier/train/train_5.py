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
embedded_sequences=Dropout(0.2)(embedded_sequences)

lstm_1 = Bidirectional(LSTM(12,return_sequences=True,dropout=0.2, recurrent_dropout=0.0))(embedded_sequences)
conv_1 = Conv1D(filters=24,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(lstm_1)
conv_1 = Dropout(0.2)(conv_1)
merge_1=Concatenate(axis=1)([lstm_1, conv_1])
lstm_2= Bidirectional(LSTM(12,return_sequences=True,dropout=0.2, recurrent_dropout=0.0))(merge_1)
conv_2 = Conv1D(filters=24,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(lstm_2)
conv_2 = Dropout(0.2)(conv_2)
merge_2=Concatenate(axis=1)([lstm_2, conv_2])

conv_3 = Conv1D(filters=24,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences)
conv_3 = Dropout(0.2)(conv_3)
lstm_3 = Bidirectional(LSTM(12,return_sequences=True,dropout=0.2, recurrent_dropout=0.0))(conv_3)
merge_3=Concatenate(axis=1)([lstm_3, conv_3])
conv_4 = Conv1D(filters=24,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(merge_3)
conv_4 = Dropout(0.2)(conv_4)
lstm_4 = Bidirectional(LSTM(12,return_sequences=True,dropout=0.2, recurrent_dropout=0.0))(conv_4)
merge_4=Concatenate(axis=1)([lstm_4, conv_4])

lstm_5 = Bidirectional(LSTM(12,return_sequences=True,dropout=0.2, recurrent_dropout=0.0))(embedded_sequences)
dense = Dense(24, activation='relu')(lstm_5)
conv_5 = Conv1D(filters=24,kernel_size=2,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences)
conv_5 = Dropout(0.2)(conv_5)

merge=Concatenate(axis=1)([merge_2, merge_4,lstm_5,conv_5])
pool= MaxPooling1D(4)(merge)

lstm_6=Bidirectional(LSTM(24,return_sequences=True,dropout=0.2, recurrent_dropout=0.0))(pool)
dense = Dense(24, activation='relu')(lstm_6)

drop= Dropout(0.3)(dense)
flat = Flatten()(drop)
dense = Dense(24, activation='relu')(flat)
preds = Dense(5, activation='softmax')(dense)

model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.000)

model_checkpoints = callbacks.ModelCheckpoint(p+"/model/checkpoint-5labels-{val_loss:.3f}.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=0)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['acc'])

print("Training Progress:")
model_log = model.fit(train_features_5,train_labels_5, validation_data=(val_features_5,val_labels_5),
          epochs=30, batch_size=200,
          callbacks=[model_checkpoints])


