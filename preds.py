#%% 

import keras
import  numpy
path  = keras.utils.get_file(
  'niettzsche.tx',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')


with  open(path, encoding='utf-8') as file:
  text = file.read().lower()

print(text[:60])
#%%
import numpy as np
maxlen=60
step = 3
sentence=[]
next_chars=[]



for i in range(0,(len(text)-maxlen), step):
  sentence.append(text[i:i+maxlen])
  next_chars.append(text[i+maxlen])
print('Number of sequence:',len(sentence))
chars = sorted(list(text))
print("Characters:", len(chars))


chars =  sorted(list(set(text)))
print("Unique  characters:",len(chars))

chars_indice = dict((char,chars.index(char)) for char in chars)

print("Vectorization...")


x = np.zeros((len(sentence),maxlen,len(chars)),dtype=np.bool)
y =  np.zeros((len(sentence),len(chars)),dtype=np.bool)

for i,sentence in enumerate(sentence):
  for t,char in enumerate(sentence):
    if  t==1 and i==1:
      print(list(sentence))
    x[i,t,chars_indice[char]]=1
  y[i,chars_indice[next_chars[i]]]=1
#%%

print(chars)