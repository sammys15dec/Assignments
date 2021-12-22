#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sohail's Random Forest Assignment


# In[2]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[3]:


# load data
company=pd.read_csv("company_data.csv")
company.head(10)


# In[4]:


company.shape


# In[5]:


company.dtypes


# In[6]:


company.info()


# In[7]:


# Converting from Categorical Data


# In[8]:


company['High'] = company.Sales.map(lambda x: 1 if x>8 else 0)


# In[9]:


company['ShelveLoc']=company['ShelveLoc'].astype('category')


# In[10]:


company['Urban']=company['Urban'].astype('category')


# In[11]:


company['US']=company['US'].astype('category')


# In[12]:


company.dtypes


# In[13]:


company.head(20)


# In[15]:


# label encoding to convert categorical values into numeric


# In[16]:


company['ShelveLoc']=company['ShelveLoc'].cat.codes


# In[17]:


company['Urban']=company['Urban'].cat.codes


# In[18]:


company['US']=company['US'].cat.codes


# In[19]:


company.head(10)


# In[20]:


company.tail(10)


# In[21]:


# Graphical Visualization


# In[22]:


sns.pairplot(company)


# In[23]:


sns.barplot(company['Sales'], company['Income'])


# In[24]:


sns.boxplot(company['Sales'], company['Income'])


# In[25]:


sns.lmplot(x='Income', y='Sales', data=company)


# In[26]:


sns.jointplot(company['Sales'], company['Income'])


# In[27]:


sns.stripplot(company['Sales'], company['Income'])


# In[28]:


sns.distplot(company['Sales'])


# In[29]:


sns.distplot(company['Income'])


# In[30]:


# Set feature and target variables


# In[31]:


feature_cols=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']


# In[32]:


x = company.drop(['Sales', 'High'], axis = 1)


# In[33]:


x = company[feature_cols]


# In[34]:


y = company.High


# In[35]:


print(x)


# In[36]:


print(y)


# In[37]:


# Splitting the data into the Training data and Test data


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[40]:


print(x_train)


# In[41]:


print(y_train)


# In[42]:


print(x_test)


# In[43]:


print(y_test)


# In[44]:


# Feature Scaling


# In[45]:


from sklearn.preprocessing import StandardScaler


# In[46]:


sc = StandardScaler()


# In[47]:


x_train = sc.fit_transform(x_train)


# In[48]:


x_test = sc.transform(x_test)


# In[49]:


print(x_train)


# In[50]:


print(x_test)


# In[52]:


# Training the Random Forest Classification model on the Training data


# In[53]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[54]:


classifier.fit(x_train, y_train)


# In[55]:


classifier.score(x_test, y_test)


# In[56]:


# Predicting the Test set results


# In[57]:


y_pred = classifier.predict(x_test)


# In[58]:


y_pred


# In[59]:


# Confusion Matrix


# In[60]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[61]:


cm = confusion_matrix(y_test, y_pred)


# In[62]:


print(cm)


# In[63]:


accuracy_score(y_test, y_pred)


# In[64]:


classifier = RandomForestClassifier(n_estimators=100, criterion='gini')
classifier.fit(x_train, y_train)


# In[65]:


classifier.score(x_test, y_test)


# In[66]:


# done

