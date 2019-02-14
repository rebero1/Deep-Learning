#%%

import random
import sys
from keras import layers
import keras
import numpy as np
import os
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text =  open(path,encoding='utf-8').read()
print(len(text))
#%%

maxlen=60
step=3

sentence=[]
next_char=[]

print(text[1:60])
print(text[60])
for i in range(0,len(text)-maxlen,step):
    sentence.append(text[i:i+maxlen])
    next_char.append(text[i+maxlen])
print("Number of sequence:",len(sentence))

chars=sorted(list(set(text)))
char_indices=dict((char,char.index(char))  for  char in chars)
x = np.zeros((len(sentence), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentence), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentence):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_char[i]]] = 1
#%%
from keras  import layers
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#%%


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
    
callable_tu=[keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2
),keras.callbacks.ModelCheckpoint('models.h5',save_best_only=True)]

 
for epoch in range(1, 60):
  print('epoch', epoch) 
  model.fit(x, y, batch_size=128, epochs=1,callbacks=callable_tu,verbose=0)
#%%
from numpy import random 
start_index = random.randint(0, len(text) - maxlen - 1) 
generated_text = text[start_index: start_index + maxlen] 
print('--- Generating with seed: "' + generated_text + '"')
for temperature in [0.2, 0.5, 1.0, 1.2]:
    print('------ temperature:', temperature)
    text_to='''
What's the deal with one of the most influential stand-ups ever to hold a mic,

    '''
    
    for i in range(400):
      sampled = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(generated_text):
        sampled[0, t, char_indices[char]] = 1.
    preds = model.predict(sampled, verbose=0)[0]
    next_index = sample(preds, temperature)
    next_char = chars[next_index]
    generated_text += next_char
    generated_text = generated_text[1:]
print(generated_text)
     
#%%


from keras.applications import inception_v3


from keras  import backend as k

k.set_learning_phase(0)


model= inception_v3.InceptionV3(weights='imagenet',
include_top=False)


layer_contribution={
    'mixed2':.2,
    'mixed3':3.,
    'mixed4':2.,
    'mixed5':1.5
}



layer_dict=dict([(layer.name,layer) for layer in model.layers])
loss = k.variable(0.)
