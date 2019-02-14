#%%
# function api


from keras.datasets import imdb
from keras import layers
import numpy as np
from keras.models import Sequential,Model

from keras  import layers
from keras import Input

input_tensor  = Input(shape=(64,))
x = layers.Dense(32,activation='relu')(input_tensor)
x=layers.Dense(32,activation='relu')(x)
output_tensor=layers.Dense(10,activation='softmax')(x)


model=Model(input_tensor,output_tensor)
model.summary()
#%%
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
import numpy as np

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy')
x_train=np.random.random((1000,64))
y_train=np.random.random((1000,10))


model.fit(x_train,y_train,epochs=5,batch_size=200)
score= model.evaluate(x_train,y_train)
score
#%%


text_vocabulary_size=10000
question_vocabulary_size=10000
answer_vocabulary_size=500



text_input=Input(shape=(None,),dtype='int32',name='text')


embedded_text=layers.Embedding(64,text_vocabulary_size)(text_input)
encoded_text=layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,),
                       dtype='int32',
                       name='question')
embedded_question = layers.Embedding(
    32, question_vocabulary_size)(question_input)


encoded_question = layers.LSTM(16)(embedded_question)
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(answer_vocabulary_size,
                      activation='softmax')(concatenated)
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
#%%


vocabulary_size=50000
num_income_group=10


posts_input=Input(shape=(None,),dtype='iint32',name='posts')
embedded_posts=layers.Embedding(256,vocabulary_size)(posts_input)
x = layers.Conv1D(128,5,activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction=layers.Dense(1,name='age')(x)
income_prediction=layers.Dense(num_income_group,activation='softmax',
name='income')(x)
gender_prediction=layers.Dense(1,activation='sigmoid',name='gender')(x)


model=Model(posts_input,
[age_prediction,income_prediction,gender_prediction])

model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], loss_weights=[0.25, 1., 10.])
#%%
from keras.layers import Conv2D,AvgPool2D
input_tensor=Input(shape=(None,))
branch_a = layers.Conv2D(32,1)(input_tensor)

branch_b = layers.Conv2D(32,1,strides=2)(input_tensor)
branch_b= layers.Conv2D(32,3,strides=2)(branch_b)

branch_c = layers.AvgPool2D(3,strides=2) (input_tensor)
branch_c = layers.Conv2D(32, 3, strides=2)(branch_c)


branch_d = layers.Conv2D(32, (1, 1), strides=2)(input_tensor)
branch_d = layers.Conv2D(64, (3, 3))(branch_d)
branch_d = layers.Conv2D(64, (3, 3), strides=2)(branch_d)

output=layers.concatenate([branch_a,branch_b,branch_c,branch_d],axis=-1)

#%%
import keras
callbacks_lists=[
  keras.callbacks.EarlyStopping(monitor='val_loss',
  patience=1),keras.callbacks.ModelCheckpoint(
    filepath='my_model.h5',
    save_best_only=True,monitor='val_loss'
  ),keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
  factor=.1,patience=1)
]
#%%

from keras.models import Sequential
from keras.preprocessing import sequence
import  keras
from keras.datasets import imdb
from keras import layers

max_features=20000
max_len=500


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train=sequence.pad_sequences(x_train,maxlen=max_len)
x_test=  sequence.pad_sequences(x_test,maxlen=max_len)


model=Sequential()
model.add(layers.Embedding(max_features,128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
#%%
callbacks=[keras.callbacks.TensorBoard(
  log_dir="my_log_dir",
  histogram_freq=1,
  embeddings_freq=1
)]


history = model.fit(x_train, y_train,
                    epochs=20, batch_size=128,verbose=0,
                    validation_split=0.2,
                    callbacks=callbacks)
l