from pickletools import optimize
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import SimpleRNN, Embedding , Dense
from tensorflow.keras. models import Sequential

max_words = 10000
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_words)

# print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
# print(f'Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}')

### MApping of words index bacl to words(for understanding)
sample_review=X_train[0]
sample_label=y_train[0]


# word_index=imdb.get_word_index()
# #word_index
# reverse_word_index = {value: key for key, value in word_index.items()}


# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sample_review])
# print(decoded_review)


# padding
max_len=200

X_train=sequence.pad_sequences(X_train,maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
# print(X_train[0])


model = Sequential()
model.add(Embedding(max_words , 128 , input_length =max_len))
model.add(SimpleRNN(128 , activation = 'tanh'))
model.add(Dense(1 , activation = 'sigmoid'))
model.build(input_shape=(None, max_len))
print(model.summary())

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )

## Create an instance of EarlyStoppping Callback
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True)

model.fit(
    X_train , y_train , epochs = 100, batch_size=32,
    validation_data = (X_test , y_test),
    callbacks = [earlystopping]
)

model.save('simple_rnn_imdb.h5')











