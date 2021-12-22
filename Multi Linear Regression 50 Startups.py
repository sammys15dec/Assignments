#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Multiple Regression


# In[2]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import math 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[3]:


#import dataset
startups = pd.read_csv('50_Startups.csv')
startups


# In[4]:


len(startups)


# In[5]:


startups.head()


# In[6]:


startups.shape


# In[7]:


#graphical representation


# In[8]:


plt.scatter(startups['Marketing Spend'], startups['Profit'], alpha=0.5)
plt.title('Scatter plot of Profit with Marketing Spend')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()


# In[9]:


plt.scatter(startups['R&D Spend'], startups['Profit'], alpha=0.5)
plt.title('Scatter plot of Profit with R&D Spend')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()


# In[10]:


plt.scatter(startups['Administration'], startups['Profit'], alpha=0.5)
plt.title('Scatter plot of Profit with Administration')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.show()


# In[ ]:


#create figure object


# In[11]:


ax=startups.groupby(['State'])['Profit'].mean().plot.bar(figsize=(12,6), fontsize=12)


# In[12]:


ax=sns.pairplot(startups)


# In[13]:


startups.State.value_counts()


# In[14]:


#Create dummy variable for States


# In[15]:


startups['New York_State']= np.where(startups['State']=='New York',1,0)


# In[16]:


startups['California_State']= np.where(startups['State']=='California',1,0)


# In[17]:


startups['Florida_State']= np.where(startups['State']=='Florida',1,0)


# In[20]:


#drop original coloumn states from dataset


# In[ ]:





# In[ ]:





# In[18]:


startups.drop(columns=['State'], axis=1, inplace=True)


# In[19]:


startups.head()


# In[21]:


dependent_variable='Profit'


# In[22]:


#create list of independent variables


# In[23]:


independent_variables=startups.columns.tolist()


# In[24]:


independent_variables.remove(dependent_variable)


# In[25]:


independent_variables


# In[27]:


#create data of independent variables


# In[26]:


X=startups[independent_variables].values
X


# In[28]:


# Create the Dependent Variables Data


# In[29]:


y=startups[dependent_variable].values
y


# In[30]:


#Split data into training set and test set


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[32]:


#transform data


# In[33]:


scaler=MinMaxScaler()


# In[34]:


X_train=scaler.fit_transform(X_train)


# In[35]:


X_test=scaler.transform(X_test)


# In[36]:


# Training the Multiple Linear Regression model on the Training set


# In[37]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[38]:


# Predicting the Test set results


# In[39]:


y_pred = regressor.predict(X_test)


# In[40]:


math.sqrt(mean_squared_error(y_test, y_pred))


# In[41]:


r2_score(y_test, y_pred)

