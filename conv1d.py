
#%%

from keras.layers import Flatten, Dense
import seaborn as sns
import numpy as  np
import matplotlib.pyplot as plt
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
max_length = 10
results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
print(results)

#%%


from keras.preprocessing.text import Tokenizer


samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts("I am a good person")

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(samples)
sequence=tokenizer.texts_to_sequences("I AM GOOD")


sequence = tokenizer.texts_to_sequences(samples)
one_hot_result= tokenizer.texts_to_matrix(sample,mode='binary')


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print(one_hot_result)

#%%

# hashing
samples = ['The cat sat on the mat.', 'The dog ate my homework.']


dimension = 1000
max_legnth=10

results = np.zeros((len(samples), max_length, dimension))
for i, sample in enumerate(samples):
  for j,word in enumerate(sample.split(' ')):
    index = abs(hash(word))%dimension
    results[i,j,index]=1
print(results)
#%%
from keras.datasets import imdb
from keras import preprocessing


max_feaures = 10000
maxlen=20


(x_train, y_train),(x_test,y_test) =imdb.load_data(num_words=max_feaures)
x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
y_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding
import  keras.activations as act
import keras.regularizers as leg
import keras.initializers as init
from keras import layers


def batch():
  return layers.BatchNormalization(axis=-1, momentum=0.99,
                                   epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                   gamma_initializer='ones', moving_mean_initializer='zeros',
                                   moving_variance_initializer='ones', beta_regularizer=None,
                                   gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)


model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen ))
 
model.add(Flatten())  
 
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,verbose=0, epochs=10,
                    batch_size=32, validation_split=0.2)

# %%


sns.set()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#%%

import os
imdb_dir = '/Users/reberoprince/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),'r',encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
#%%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen=100
training_samples=200
validation_samples=10000
max_words=10000


tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequence=tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("Found %s unique token"%len(word_index))
data= pad_sequences(sequence,maxlen=maxlen)



labels=np.asarray(labels)


indice=np.arange(data.shape[0])
np.random.shuffle(indice)
data=data[indice]
labels=labels[indice]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples] 
y_val = labels[training_samples: training_samples + validation_samples]
#%%
one_hot_result = tokenizer.texts_to_matrix("I am doing great", mode="binary")
 

tokenizer=Tokenizer(num_words=20)
tokenizer.fit_on_texts("bad")
arrra = tokenizer.texts_to_matrix("This is not good")
arrra= preprocessing.sequence.pad_sequences(arrra,maxlen=20)

model.predict_classes(arrra)