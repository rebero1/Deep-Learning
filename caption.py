#%%
from  keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.preprocessing import image
import os
import numpy as np

from keras.utils import  plot_model

def predict(path,variable=1):
  if(not os.path.isfile(path)):
   raise Exception(" File not Found")
  image_format =['jpeg','png','jpg']
 
  assert(path[-4:] in image_format or path[-3:] in image_format),"format not known"
  model = VGG16(weights='imagenet')
  plot_model(model,to_file='VGG16.png',show_layer_names=True,show_shapes=True)
  
  img = image.load_img(path,target_size=(224,224))
  x = image.img_to_array(img)
  x = np.expand_dims(x,axis=0)
  x = preprocess_input(x)


  preds=model.predict(x)
  predicted = decode_predictions(preds, top=variable)
  print("Predicted:",predicted[0][0][1])


path = "/Users/reberoprince/Desktop/kk.png"
predict(path=path,variable=1)
