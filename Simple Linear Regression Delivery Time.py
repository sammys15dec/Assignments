#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Simple Linear Regression


# In[2]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew


# In[3]:


#load dataset
data1=pd.read_csv("delivery_time.csv")
data1


# In[4]:


#analyze data


# In[5]:


data1.head()


# In[6]:


data1.describe()


# In[7]:


data1.info()


# In[8]:


data1.shape


# In[9]:


data1 = data1.rename(columns = {'Delivery Time': 'DT', 'Sorting Time': 'ST'}, inplace = False)
data1.info()


# In[10]:


print(kurtosis(data1.DT))


# In[11]:


print(kurtosis(data1.ST))


# In[12]:


print(skew(data1.DT))


# In[13]:


print(skew(data1.ST))


# In[14]:


#graphical representation


# In[15]:


data1.plot()


# In[16]:


sns.pairplot(data1)


# In[17]:


data1.corr()


# In[18]:


sns.distplot(data1['DT'])


# In[19]:


sns.distplot(data1['ST'])


# In[20]:


corrMatrix = data1.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[28]:


colms = data1.columns 
colours = ['#ffc0cb', '#ffff00']
sns.heatmap(data1[colms].isnull(),
            cmap=sns.color_palette(colours))


# In[29]:


data1.boxplot(column=['DT'])


# In[30]:


data1.boxplot(column=['ST'])


# In[31]:


data1[data1.duplicated()].shape


# In[32]:


data1['ST'].hist()


# In[33]:


data1.boxplot(column=['ST'])


# In[34]:


data1['ST'].value_counts().plot.bar()


# In[35]:


#calculating R^2 value


# In[36]:


import statsmodels.formula.api as smf
model = smf.ols("DT~ST",data = data1).fit()
model


# In[37]:


sns.regplot(x="ST", y="DT", data=data1);


# In[38]:


model.params


# In[39]:


print(model.tvalues, '\n', model.pvalues)


# In[40]:


(model.rsquared,model.rsquared_adj)


# In[41]:


model.summary()


# In[42]:


data_1=data1
data_1['DT'] = np.log(data_1['DT'])
data_1['ST'] = np.log(data_1['ST'])
sns.distplot(data_1['DT'])
fig = plt.figure()
sns.distplot(data_1['ST'])
fig = plt.figure()


# In[43]:


model_2 = smf.ols("ST~DT",data = data_1).fit()
model_2.summary()


# In[44]:


data_2=data1
data_1['DT'] = np.log(data_1['DT'])
sns.distplot(data_1['DT'])
fig = plt.figure()
sns.distplot(data_1['ST'])
fig = plt.figure()


# In[45]:


model_3 = smf.ols("ST~DT",data = data_2).fit()
model_3.summary()


# In[46]:


data_3=data1
data_1['ST'] = np.log(data_1['ST'])
sns.distplot(data_1['DT'])
fig = plt.figure()
sns.distplot(data_1['ST'])
fig = plt.figure()


# In[47]:


model_4 = smf.ols("ST~DT",data = data_3).fit()
model_4.summary()


# In[49]:


import statsmodels.formula.api as smf
import numpy as np
import pandas.testing as tm
model_4 = smf.ols("DT~ST",data = data_3).fit()


# In[50]:


#predict for new data point


# In[51]:


#predict for 15 and 20 min sorting time
newdata=pd.Series([15,20])


# In[52]:


data_pred=pd.DataFrame(newdata,columns=['ST'])
data_pred


# In[53]:


model_4.predict(data_pred)


# In[ ]:





# In[54]:


model_4

