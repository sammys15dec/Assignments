#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's KNN Assignment


# In[2]:


#import libraries
from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


#load data
data = read_csv("glass.csv")


# In[4]:


data.head(10)


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


# Preprocessing 


# In[10]:


array = data.values
X = array[:, 0:9]
X


# In[11]:


Y = array[:, 9]
Y


# In[12]:


kfold = KFold(n_splits=10)


# In[13]:


model = KNeighborsClassifier(n_neighbors=15)
results = cross_val_score(model, X, Y, cv=kfold)


# In[14]:


print(results.mean())


# In[15]:


# Grid Search for Algorithm Tuning


# In[16]:


import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[17]:


n_neighbors1 = numpy.array(range(1,80))
param_grid = dict(n_neighbors=n_neighbors1)


# In[18]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[19]:


print(grid.best_score_)


# In[20]:


print(grid.best_params_)


# In[21]:


# Visualize the CV results


# In[22]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 80
k_range = range(1, 80)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[23]:


#done

