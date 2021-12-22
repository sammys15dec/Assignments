#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sohail's Neural Network Assignment


# In[2]:


# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


# load data
data_raw = pd.read_csv("gas_turbines.csv")
data_raw


# In[6]:


data_raw.info()


# In[7]:


data_raw.describe()


# In[8]:


for i in data_raw.columns:
    print(i)
    sns.boxplot(data_raw[i])
    plt.show()


# In[9]:


for i in data_raw.columns:
    plt.scatter(np.log(data_raw[i]), np.log(data_raw['TEY']))
    plt.title(i+ ' vs TEY')
    plt.grid()
    plt.show()


# In[10]:


sns.pairplot(data_raw)


# In[11]:


data = data_raw[data_raw['CO']<data_raw['CO'].quantile(.9)]


# In[12]:


sns.boxplot(data['CO'])


# In[13]:


X = data_raw.drop(columns = ['TEY'], axis = 1) 
y = data_raw.iloc[:,7]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)
X_test_scaled


# In[15]:


input_size = len(X.columns)
output_size = 1
hidden_layer_size = 50

model = tf.keras.Sequential([
                                
                               tf.keras.layers.Dense(hidden_layer_size, input_dim = input_size, activation = 'relu'),
                               tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                               tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                               tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),     
                               tf.keras.layers.Dense(output_size)
    
    
                                ])


# In[16]:


optimizer = tf.keras.optimizers.SGD(learning_rate = 0.03)
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['MeanSquaredError'])


# In[17]:


num_epochs = 50
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)
model.fit(X_train_scaled, y_train, callbacks = early_stopping, validation_split = 0.1, epochs = num_epochs, verbose = 2)


# In[18]:


test_loss, mean_squared_error = model.evaluate(X_test_scaled, y_test)


# In[19]:


predictions = model.predict_on_batch(X_test_scaled)


# In[20]:


plt.scatter(y_test, predictions)


# In[21]:


predictions_df = pd.DataFrame()
predictions_df['Actual'] = y_test
predictions_df['Predicted'] = predictions
predictions_df['% Error'] = abs(predictions_df['Actual'] - predictions_df['Predicted'])/predictions_df['Actual']*100
predictions_df.reset_index(drop = True)


# In[22]:


# done

