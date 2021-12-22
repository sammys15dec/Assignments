#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Naive Bayes Assignment


# In[2]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#load data


# In[9]:


salary_train=pd.read_csv('SalaryData_Train.csv')
salary_train


# In[10]:


salary_test=pd.read_csv('SalaryData_Test.csv')
salary_test


# In[6]:


salary_train.columns


# In[7]:


salary_test.columns


# In[11]:


salary_train.dtypes


# In[12]:


salary_test.dtypes


# In[13]:


salary_train.info()


# In[14]:


salary_test.info()


# In[16]:


string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[17]:


# Graphical Visualization


# In[18]:


sns.pairplot(salary_train)


# In[19]:


sns.pairplot(salary_test)


# In[20]:


sns.boxplot(salary_train['Salary'], salary_train['capitalgain'])


# In[21]:


sns.boxplot(salary_test['Salary'], salary_test['capitalgain'])


# In[22]:


sns.countplot(salary_train['Salary'])


# In[23]:


sns.countplot(salary_test['Salary'])


# In[24]:


plt.figure(figsize=(20,10))
sns.barplot(x='Salary', y='hoursperweek', data=salary_train)
plt.show()


# In[25]:


plt.figure(figsize=(20,10))
sns.barplot(x='Salary', y='hoursperweek', data=salary_test)
plt.show()


# In[26]:


sns.distplot(salary_train['hoursperweek'])


# In[27]:


sns.distplot(salary_test['hoursperweek'])


# In[28]:


plt.figure(figsize=(15,10))
sns.lmplot(y='capitalgain', x='hoursperweek',data=salary_train)
plt.show()


# In[29]:


plt.figure(figsize=(15,10))
sns.lmplot(y='capitalgain', x='hoursperweek',data=salary_test)
plt.show()


# In[30]:


sns.factorplot(x='Salary', hue='hoursperweek', data=salary_train, kind='count', height=10,aspect=1.5)


# In[31]:


sns.factorplot(x='Salary', hue='hoursperweek', data=salary_test, kind='count', height=10,aspect=1.5)


# In[32]:


sns.catplot(x='Salary', hue='hoursperweek', data=salary_train, kind='count',aspect=1.5)


# In[33]:


sns.catplot(x='Salary', hue='hoursperweek', data=salary_test, kind='count',aspect=1.5)


# In[35]:


plt.subplots(1,2, figsize=(16,8))

colors = ["#FF0000", "#64FE2E"]
labels ="capitalgain", "capitalloss"

plt.suptitle('salary of an individual', fontsize=18)

salary_train["Salary"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, colors=colors, 
                                             labels=labels, fontsize=10, startangle=25)


# In[36]:


plt.style.use('seaborn-whitegrid')

salary_train.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# In[37]:


plt.subplots(1,2, figsize=(16,8))

colors = ["#FF0000", "#64FE2E"]
labels ="capitalgain", "capitalloss"

plt.suptitle('salary of an individual', fontsize=18)

salary_test["Salary"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, colors=colors, 
                                             labels=labels, fontsize=10, startangle=25)


# In[38]:


plt.style.use('seaborn-whitegrid')

salary_test.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# In[39]:


# Preprocessing


# In[40]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[41]:


for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])


# In[42]:


col_names=list(salary_train.columns)
col_names


# In[43]:


train_X=salary_train[col_names[0:13]]
train_X


# In[44]:


train_Y=salary_train[col_names[13]]
train_Y


# In[45]:


test_x=salary_test[col_names[0:13]]
test_x


# In[46]:


test_y=salary_test[col_names[13]]
test_y


# In[47]:


# Build Naive Bayes Model


# In[48]:


# Gaussian Naive Bayes


# In[49]:


from sklearn.naive_bayes import GaussianNB
Gnbmodel=GaussianNB()


# In[50]:


train_pred_gau=Gnbmodel.fit(train_X,train_Y).predict(train_X)
train_pred_gau


# In[51]:


test_pred_gau=Gnbmodel.fit(train_X,train_Y).predict(test_x)
test_pred_gau


# In[52]:


train_acc_gau=np.mean(train_pred_gau==train_Y)


# In[53]:


test_acc_gau=np.mean(test_pred_gau==test_y)


# In[54]:


train_acc_gau


# In[55]:


test_acc_gau


# In[56]:


# Multinomial Naive Bayes


# In[57]:


from sklearn.naive_bayes import MultinomialNB
Mnbmodel=MultinomialNB()


# In[58]:


train_pred_multi=Mnbmodel.fit(train_X,train_Y).predict(train_X)
train_pred_multi


# In[59]:


test_pred_multi=Mnbmodel.fit(train_X,train_Y).predict(test_x)
test_pred_multi


# In[60]:


train_acc_multi=np.mean(train_pred_multi==train_Y)
train_acc_multi


# In[61]:


test_acc_multi=np.mean(test_pred_multi==test_y)
test_acc_multi


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




