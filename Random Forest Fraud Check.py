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
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[3]:


#load data
fraud=pd.read_csv("Fraud_check.csv")
fraud.head(20)


# In[4]:


fraud.dtypes


# In[5]:


fraud.info()


# In[6]:


fraud.columns


# In[7]:


fraud.shape


# In[8]:


fraud.isnull().sum()


# In[9]:


fraud["TaxInc"] = pd.cut(fraud["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
fraud["TaxInc"]


# In[10]:


fraudcheck = fraud.drop(columns=["Taxable.Income"])
fraudcheck 


# In[11]:


FC = pd.get_dummies(fraudcheck .drop(columns = ["TaxInc"]))


# In[12]:


Fraud_final = pd.concat([FC,fraudcheck ["TaxInc"]], axis = 1)


# In[13]:


colnames = list(Fraud_final.columns)
colnames


# In[14]:


predictors = colnames[:9]
predictors


# In[15]:


target = colnames[9]
target


# In[16]:


X = Fraud_final[predictors]
X.shape


# In[17]:


Y = Fraud_final[target]
Y


# In[18]:


# Graphical Visualization


# In[19]:


sns.pairplot(fraud)


# In[20]:


sns.barplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[21]:


sns.boxplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[22]:


sns.lmplot(x='Taxable.Income',y='City.Population', data=fraud)


# In[23]:


sns.jointplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[24]:


sns.stripplot(fraud['Taxable.Income'], fraud['City.Population'])


# In[25]:


sns.distplot(fraud['Taxable.Income'])


# In[26]:


sns.distplot(fraud['City.Population'])


# In[27]:


# Building Random Forest Model


# In[28]:


from sklearn.ensemble import RandomForestClassifier


# In[29]:


rf = RandomForestClassifier(n_jobs = 3, oob_score = True, n_estimators = 15, criterion = "entropy")


# In[30]:


rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[31]:


np.shape(Fraud_final) 


# In[32]:


Fraud_final.describe()


# In[33]:


Fraud_final.info()


# In[34]:


type([X])


# In[35]:


type([Y])


# In[36]:


Y1 = pd.DataFrame(Y)
Y1


# In[37]:


type(Y1)


# In[38]:


rf.fit(X,Y1) 


# In[39]:


rf.estimators_ 


# In[40]:


rf.classes_ 


# In[41]:


rf.n_classes_


# In[42]:


rf.n_features_ 


# In[43]:


rf.n_outputs_ 


# In[44]:


rf.oob_score_


# In[45]:


rf.predict(X)


# In[46]:


Fraud_final['rf_pred'] = rf.predict(X)


# In[47]:


cols = ['rf_pred','TaxInc']


# In[48]:


Fraud_final[cols].head()


# In[49]:


Fraud_final["TaxInc"]


# In[50]:


# Confusion Matrix


# In[51]:


from sklearn.metrics import confusion_matrix


# In[53]:


confusion_matrix(Fraud_final['TaxInc'],Fraud_final['rf_pred']) 


# In[54]:


pd.crosstab(Fraud_final['TaxInc'],Fraud_final['rf_pred'])


# In[55]:


print("Accuracy",(476+115)/(476+115+9+0)*100)


# In[56]:


Fraud_final["rf_pred"]


# In[57]:


# done

