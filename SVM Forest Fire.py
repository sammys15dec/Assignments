#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sohail's SVM Assignment


# In[2]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[3]:


# load data
fire= pd.read_csv("forestfires.csv")
fire.head(20)


# In[4]:


# Preprocessing & Label Encoding 


# In[5]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
fire["month"] = label_encoder.fit_transform(fire["month"])
fire["day"] = label_encoder.fit_transform(fire["day"])
fire["size_category"] = label_encoder.fit_transform(fire["size_category"])


# In[6]:


fire.head(10)


# In[7]:


# Graphical Visualization


# In[8]:


for i in fire.describe().columns[:-2]:
    fire.plot.scatter(i,'area',grid=True)


# In[9]:


fire.groupby('day').area.mean().plot(kind='bar')


# In[10]:


fire.groupby('day').area.mean().plot(kind='box')


# In[11]:


fire.groupby('month').area.mean().plot(kind='box')


# In[12]:


fire.groupby('day').area.mean().plot(kind='line')


# In[13]:


fire.groupby('day').area.mean().plot(kind='hist')


# In[14]:


fire.groupby('day').area.mean().plot(kind='density')


# In[15]:


X=fire.iloc[:,:11]
X


# In[16]:


y=fire["size_category"]
y


# In[17]:


# Split the Data intp Training Data and Test Data


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# In[19]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[20]:


# Grid Search CV


# In[21]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[22]:


gsv.best_params_ , gsv.best_score_


# In[23]:


clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[24]:


clf1 = SVC(C= 15, gamma = 50)
clf1.fit(X , y)
y_pred = clf1.predict(X)
acc1 = accuracy_score(y, y_pred) * 100
print("Accuracy =", acc1)
confusion_matrix(y, y_pred)


# In[25]:


# Poly


# In[26]:


clf2 = SVC()
param_grid = [{'kernel':['poly'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[27]:


gsv.best_params_ , gsv.best_score_


# In[28]:


# Sigmoid


# In[29]:


clf3 = SVC()
param_grid = [{'kernel':['sigmoid'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[30]:


gsv.best_params_ , gsv.best_score_


# In[31]:


# done

