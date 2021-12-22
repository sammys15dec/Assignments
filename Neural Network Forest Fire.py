#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sohail's Neural Netwrk Assignment


# In[2]:


#import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
import numpy


# In[3]:


import pandas as pd
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[4]:


#load data
dataset = pd.read_csv("forestfires.csv")
dataset.head(20)


# In[5]:


dataset.shape


# In[6]:


dataset.dtypes


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# In[9]:


# Label Encoding


# In[11]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
dataset["month"] = label_encoder.fit_transform(dataset["month"])
dataset["day"] = label_encoder.fit_transform(dataset["day"])
dataset["size_category"] = label_encoder.fit_transform(dataset["size_category"])


# In[12]:


dataset.head(10)


# In[13]:


# split into input (X) and output (Y) variables


# In[14]:


X = dataset.iloc[:,:11]


# In[15]:


Y = dataset.iloc[:,-1]


# In[16]:


X


# In[17]:


Y


# In[18]:


# Build ANN Model


# In[19]:


model = Sequential()
model.add(layers.Dense(50, input_dim=11,  activation='relu'))
model.add(layers.Dense(11,  activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[20]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# In[21]:


# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=10)


# In[22]:


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[23]:


# Graphical Visualization


# In[24]:


history.history.keys()


# In[25]:


# summarize history for accuracy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[26]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[27]:


# done

