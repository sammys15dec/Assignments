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
train= pd.read_csv("SalaryData_Train(1).csv")
train.head()


# In[4]:


test=pd.read_csv("SalaryData_Test(1).csv")
test.head()


# In[5]:


train.shape


# In[6]:


test.shape


# In[7]:


# Preprocessing and Encoding


# In[8]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
train["workclass"] = label_encoder.fit_transform(train["workclass"])
train["education"] = label_encoder.fit_transform(train["education"])
train["maritalstatus"] = label_encoder.fit_transform(train["maritalstatus"])
train["occupation"] = label_encoder.fit_transform(train["occupation"])
train["relationship"] = label_encoder.fit_transform(train["relationship"])
train["race"] = label_encoder.fit_transform(train["race"])
train["sex"] = label_encoder.fit_transform(train["sex"])
train["native"] = label_encoder.fit_transform(train["native"])
train["Salary"] = label_encoder.fit_transform(train["Salary"])


# In[9]:


test["workclass"] = label_encoder.fit_transform(test["workclass"])
test["education"] = label_encoder.fit_transform(test["education"])
test["maritalstatus"] = label_encoder.fit_transform(test["maritalstatus"])
test["occupation"] = label_encoder.fit_transform(test["occupation"])
test["relationship"] = label_encoder.fit_transform(test["relationship"])
test["race"] = label_encoder.fit_transform(test["race"])
test["sex"] = label_encoder.fit_transform(test["sex"])
test["native"] = label_encoder.fit_transform(test["native"])
test["Salary"] = label_encoder.fit_transform(test["Salary"])


# In[10]:


train


# In[11]:


test


# In[12]:


# Graphical Visualization


# In[13]:


train.groupby('education').Salary.mean().plot(kind='bar')


# In[14]:


test.groupby('education').Salary.mean().plot(kind='box')


# In[15]:


train.groupby('education').Salary.mean().plot(kind='line')


# In[16]:


test.groupby('education').Salary.mean().plot(kind='hist')


# In[17]:


train.groupby('education').Salary.mean().plot(kind='density')


# In[18]:


X_train=train.iloc[:,:-1]
X_train


# In[19]:


y_train=train.iloc[:,-1]
y_train


# In[20]:


X_test=test.iloc[:,:-1]
X_test


# In[21]:


y_test=test.iloc[:,-1]
y_test


# In[22]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[23]:


# Grid Search CV


# In[24]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[5],'C':[15] }]
gsv = GridSearchCV(clf,param_grid,cv=3)
gsv.fit(X_train,y_train)


# In[25]:


gsv.best_params_ , gsv.best_score_


# In[26]:


clf = SVC(C= 15, gamma = 5)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[27]:


salary = pd.merge(train,test)


# In[28]:


salary


# In[29]:


X=salary.iloc[:,:-1]
X


# In[30]:


y=salary.iloc[:,-1]
y


# In[31]:


clf1 = SVC(C= 15, gamma = 50)
clf1.fit(X , y)
y_pred = clf1.predict(X)
acc1 = accuracy_score(y, y_pred) * 100
print("Accuracy =", acc1)
confusion_matrix(y, y_pred)


# In[32]:


# dome


# In[ ]:





# In[ ]:





# In[ ]:




