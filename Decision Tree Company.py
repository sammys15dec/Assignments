#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Sohail's Decision Tree


# In[3]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree 
from sklearn.metrics import classification_report 
from sklearn import preprocessing


# In[5]:


#load data
Company_data= pd.read_csv("Company_Data.csv")
Company_data.head(20)


# In[6]:


Company_data.shape


# In[7]:


Company_data.info()


# In[8]:


Company_data.dtypes


# In[9]:


Company_data.describe()


# In[10]:


Company_data.corr()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(Company_data)


# In[13]:


Comp_Data = Company_data


# In[14]:


Comp_Data


# In[15]:


sns.barplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[16]:


sns.boxplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[17]:


sns.lmplot(x='Income', y='Sales', data=Comp_Data)


# In[18]:


sns.jointplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[19]:


sns.swarmplot(Comp_Data['Sales'], Comp_Data['Income'])


# In[20]:


sns.distplot(Comp_Data['Sales'])


# In[21]:


sns.distplot(Comp_Data['Income'])


# In[22]:


# Preprocessing


# In[23]:


Comp_Data.loc[Comp_Data["Sales"] <= 10.00,"Sales1"]="Not High"
Comp_Data.loc[Comp_Data["Sales"] >= 10.01,"Sales1"]="High"


# In[24]:


Comp_Data


# In[25]:


# Label Encoding


# In[26]:


label_encoder = preprocessing.LabelEncoder()
Comp_Data["ShelveLoc"] = label_encoder.fit_transform(Comp_Data["ShelveLoc"])
Comp_Data["Urban"] = label_encoder.fit_transform(Comp_Data["Urban"])
Comp_Data["US"] = label_encoder.fit_transform(Comp_Data["US"])
Comp_Data["Sales1"] = label_encoder.fit_transform(Comp_Data["Sales1"])


# In[27]:


Comp_Data


# In[28]:


x=Comp_Data.iloc[:,1:11]
x


# In[29]:


y=Comp_Data["Sales1"]
y


# In[30]:


Comp_Data.Sales1.value_counts()


# In[31]:


colnames=list(Comp_Data.columns)
colnames


# In[32]:


# Split Data into Train data and Test data


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[34]:


model = DecisionTreeClassifier(criterion = 'entropy')


# In[35]:


model.fit(x_train,y_train)


# In[36]:


tree.plot_tree(model);


# In[38]:


fn=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']
cn=['Not High Sales', 'High Sales']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,8), dpi=800)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[39]:


preds=model.predict(x_test)
pd.Series(preds).value_counts()


# In[40]:


pd.Series(y_test).value_counts()


# In[41]:


pd.crosstab(y_test,preds)


# In[42]:


np.mean(preds==y_test)


# In[43]:


# Building Decision Tree Classifier (CART) using Gini Criteria


# In[44]:


model_gini = DecisionTreeClassifier(criterion='gini')


# In[45]:


model_gini.fit(x_train, y_train)


# In[46]:


#Prediction and computing the accuracy


# In[47]:


pred=model.predict(x_test)
np.mean(preds==y_test)


# In[48]:


# Decision Tree Regression


# In[49]:


array=Comp_Data.values


# In[50]:


X=array[:,1:11]
X


# In[51]:


y=array[:,-1]
y


# In[52]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


# In[53]:


from sklearn.tree import DecisionTreeRegressor
model1=DecisionTreeRegressor()


# In[54]:


model1.fit(X_train, y_train)


# In[55]:


# Finding accuracy


# In[56]:


model1.score(X_test,y_test)


# In[57]:


# this data is not good for Decision tree Regression


# In[ ]:




