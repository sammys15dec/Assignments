#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sohail's Decision Tree Assignment


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


# In[4]:


# load data
Fraud_check= pd.read_csv("Fraud_check.csv")
Fraud_check.head(20)


# In[5]:


Fraud_check.shape


# In[6]:


Fraud_check.dtypes


# In[7]:


Fraud_check.info()


# In[8]:


Fraud_check.describe()


# In[9]:


Fraud_check.corr()


# In[10]:


Fraud_check.columns


# In[11]:


Fraud_check.isnull().sum()


# In[12]:


# Graphical Visualization


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(Fraud_check)


# In[14]:


sns.barplot(Fraud_check['Taxable.Income'], Fraud_check['City.Population'])


# In[15]:


sns.boxplot(Fraud_check['Taxable.Income'], Fraud_check['City.Population'])


# In[16]:


sns.lmplot(x='Taxable.Income',y='City.Population', data=Fraud_check)


# In[17]:


sns.jointplot(Fraud_check['Taxable.Income'], Fraud_check['City.Population'])


# In[18]:


sns.stripplot(Fraud_check['Taxable.Income'], Fraud_check['City.Population'])


# In[19]:


sns.distplot(Fraud_check['Taxable.Income'])


# In[20]:


sns.distplot(Fraud_check['City.Population'])


# In[21]:


# Preprocessing


# In[22]:


Fraud_check.loc[Fraud_check["Taxable.Income"] <= 30000,"Taxable_Income"]="Good"
Fraud_check.loc[Fraud_check["Taxable.Income"] > 30001,"Taxable_Income"]="Risky"
#Fraud_check.loc[Fraud_check["Taxable.Income"]!="Good","Taxable_Income"]="Risky"


# In[23]:


Fraud_check


# In[24]:


# Label Encoding


# In[25]:


label_encoder = preprocessing.LabelEncoder()
Fraud_check["Undergrad"] = label_encoder.fit_transform(Fraud_check["Undergrad"])
Fraud_check["Marital.Status"] = label_encoder.fit_transform(Fraud_check["Marital.Status"])
Fraud_check["Urban"] = label_encoder.fit_transform(Fraud_check["Urban"])
Fraud_check["Taxable_Income"] = label_encoder.fit_transform(Fraud_check["Taxable_Income"])


# In[26]:


Fraud_check.drop(['City.Population'],axis=1,inplace=True)
Fraud_check.drop(['Taxable.Income'],axis=1,inplace=True)


# In[27]:


Fraud_check["Taxable_Income"].unique()


# In[28]:


Fraud_check


# In[29]:


x = Fraud_check.iloc[:,0:4]
x


# In[30]:


y = Fraud_check["Taxable_Income"]
y


# In[31]:


len(y)


# In[32]:


colnames=list(Fraud_check.columns)
colnames


# In[33]:


# Split into Train data and Test Data


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)


# In[35]:


model=DecisionTreeClassifier(criterion="gini")
model.fit(x_train,y_train)


# In[36]:


# Build Decision Tree


# In[39]:


fn=[ 'Undergrad',
 'Marital.Status',
 'Taxable.Income',
 'Work.Experience',
   'Urban']
cn=['Good','Risky']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,8), dpi=800)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[40]:


preds=model.predict(x_test)
pd.Series(preds).value_counts()


# In[41]:


pd.Series(y_test).value_counts()


# In[42]:


pd.crosstab(y_test,preds)


# In[43]:


np.mean(preds==y_test)


# In[44]:


array=Fraud_check.values
array


# In[45]:


X=array[:,0:4]
X


# In[46]:


Y=array[:,4]
Y


# In[47]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)


# In[48]:


from sklearn.tree import DecisionTreeRegressor
model1=DecisionTreeRegressor()


# In[49]:


model1.fit(X_train, Y_train)


# In[50]:


model1.score(X_test, Y_test)


# In[51]:


# The Regressor method is not best fit for Decision Tree

