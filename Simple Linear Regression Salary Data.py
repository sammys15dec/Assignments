#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohails's Simple Linear Regression


# In[2]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns


# In[3]:


#load dataset
data=pd.read_csv("Salary_Data.csv")
data


# In[4]:


#analyze data


# In[5]:


data.head()


# In[6]:


data.info()


# In[9]:


#Graphical Representation


# In[10]:


data.plot()


# In[11]:


data.corr()


# In[12]:


data.Salary


# In[13]:


data.YearsExperience


# In[14]:


sns.distplot(data['Salary'])


# In[15]:


sns.distplot(data['YearsExperience'])


# In[16]:


sns.pairplot(data)


# In[17]:


sns.scatterplot(x=data.YearsExperience, y=np.log(data.Salary), data=data)


# In[18]:


# Calculate R^2 values


# In[20]:


import statsmodels.formula.api as smf
import pandas.testing as tm
model = smf.ols("Salary~YearsExperience",data = data).fit()


# In[21]:


sns.regplot(x="Salary", y="YearsExperience", data=data);


# In[22]:


#Coefficients
model.params


# In[23]:


model =smf.ols('Salary~YearsExperience', data=data).fit()
model


# In[24]:


model.summary()


# In[25]:


# Predict for new data point


# In[26]:


#Predict for 15 and 20 Year's of Experiance 
newdata=pd.Series([15,20])


# In[27]:


data_pred=pd.DataFrame(newdata,columns=['YearsExperience'])
data_pred


# In[28]:


model.predict(data_pred).round(2)

