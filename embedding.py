from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import pad_sequences
import numpy as np

### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

voc_size = 10000

# ohe
oh_repr = [one_hot(s , voc_size) for s in sent]
print(oh_repr)

word_vec_dim = 10

# padding
sent_length = 8
embedded_docs=pad_sequences(oh_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# model building
model=Sequential()
model.add(Embedding(voc_size,word_vec_dim,input_length=sent_length))
model.compile('adam','mse')
model.build(input_shape=(None, sent_length))
print(model.summary())


# train model menas conver each word into dense vector of 10 dim
emb = model.predict(embedded_docs)
print(emb)

 
