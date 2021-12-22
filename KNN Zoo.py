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
Zoo = read_csv("Zoo.csv")
Zoo.head(10)


# In[4]:


Zoo.shape


# In[5]:


Zoo.dtypes


# In[6]:


Zoo.info()


# In[7]:


Zoo.describe()


# In[8]:


# Preprocessing


# In[9]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
Zoo["animal name"] = label_encoder.fit_transform(Zoo["animal name"])


# In[10]:


Zoo


# In[11]:


array = Zoo.values
X = array[:, 1:17]
X


# In[12]:


Y = array[:, -1]
Y


# In[13]:


kfold = KFold(n_splits=4)


# In[14]:


model = KNeighborsClassifier(n_neighbors=13)
results = cross_val_score(model, X, Y, cv=kfold)


# In[15]:


print(results.mean())


# In[17]:


# Grid Search for Algorithm Tuning


# In[16]:


import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[21]:


n_neighbors1 = numpy.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors1)


# In[22]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[23]:


print(grid.best_score_)


# In[24]:


print(grid.best_params_)


# In[25]:


#visualize the CV results


# In[26]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 70
k_range = range(1, 70)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=4)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[28]:


#done

