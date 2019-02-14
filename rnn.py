#%%
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.models import Sequential from keras import layers
import os

data_dir = '/Users/reberoprince/Downloads'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

with open(fname) as f:
  data = f.read()
lines=data.split('\n')
header=lines[0].split(',')
lines=lines[1:]

print(header)
print(len(lines))
#%%
import numpy as np

float_data=np.zeros((len(lines),len(header)-1))



for pos,line in enumerate(lines):
  value  =[float(x) for x in line.split(',')[1:]]
  float_data[pos,:]=value
#%%

import matplotlib.pyplot as  plt
import seaborn as sns 
sns.set()
temp=float_data[:,1] 
plt.plot(range(len(temp)),temp)
plt.show()


#%%
# plot for ten days

 
plt.plot(range(len(temp[:1440])), temp[:1440],'g.')

#%%
# preprocessing


mean = float_data[:20000].mean(axis=0)
float_data-=mean
std = float_data[:20000].std(axis=0)
float_data/=std


#%%

#generator

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
#%%

lookback=1440
step=6
delay =144
batch_size=14

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)
                    
val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)
#%%


def plot_graph(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()

#%%
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1]))) 
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae') 
history = model.fit_generator(train_gen,
 steps_per_epoch=500, epochs=20,verbose=0, validation_data=val_gen, validation_steps=val_steps)
#%%

plot_graph(history)
#%%
from keras.preprocessing  import sequence

from keras.datasets  import imdb


max_features=10000
maxlen=500
(x_train,y_train),(x_test,y_test)  = imdb.load_data(num_words=max_features)
x_train=sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
#%%
max_len=maxlen
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
