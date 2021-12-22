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
import warnings
import os
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[3]:


#load data
Toyota = pd.read_csv("ToyotaCorolla.csv",encoding='ISO-8859-1')
Toyota


# In[4]:


Toyota1= Toyota.iloc[:,[2,3,6,8,12,13,15,16,17]]
Toyota1


# In[7]:


Toyota1.rename(columns={"Age_08_04":"Age"},inplace=True)


# In[8]:


eda=Toyota1.describe()
eda


# In[9]:


#data visualization


# In[10]:


plt.boxplot(Toyota1["Price"])


# In[11]:


plt.boxplot(Toyota1["Age"])


# In[12]:


plt.boxplot(Toyota1["HP"])


# In[13]:


plt.boxplot(Toyota1["cc"])


# In[14]:


plt.boxplot(Toyota1["Quarterly_Tax"])


# In[15]:


plt.boxplot(Toyota1["Weight"])


# In[16]:


plt.scatter(Toyota1['Age'], Toyota1['Price'], c = 'red')
plt.title('Price vs Age of the Cars')
plt.xlabel('Age in Years')
plt.ylabel('Price(Euros)')
plt.show()


# In[17]:


plt.figure(figsize=(8,8))
plt.title('Car Price Distribution Plot')
sns.distplot(Toyota1['Price'])


# In[18]:


import statsmodels.api as sm


# In[19]:


sm.graphics.qqplot(Toyota1["Price"])


# In[20]:


sm.graphics.qqplot(Toyota1["Age"])


# In[21]:


sm.graphics.qqplot(Toyota1["HP"])


# In[22]:


sm.graphics.qqplot(Toyota1["Quarterly_Tax"])


# In[23]:


sm.graphics.qqplot(Toyota1["Weight"])


# In[24]:


sm.graphics.qqplot(Toyota1["Gears"])


# In[25]:


sm.graphics.qqplot(Toyota1["Doors"])


# In[26]:


sm.graphics.qqplot(Toyota1["cc"])


# In[27]:


plt.hist(Toyota1["Price"])


# In[28]:


plt.hist(Toyota1["Age"])


# In[29]:


plt.hist(Toyota1["HP"])


# In[30]:


plt.hist(Toyota1["Quarterly_Tax"])


# In[31]:


plt.hist(Toyota1["Weight"])


# In[32]:


sns.pairplot(Toyota1)


# In[33]:


plt.hist(Toyota1['KM'], edgecolor = 'white', bins = 5)
plt.title('Histogram of Kilometer')
plt.xlabel('Kilometer')
plt.ylabel('Frequency')
plt.show()


# In[34]:


plt.figure(figsize=(20, 6))
plt.hist(Toyota1['KM'],facecolor ="peru",edgecolor ="blue",bins =100)
plt.ylabel("Frequency");
plt.xlabel(" Total KM")
plt.show()


# In[35]:


plt.figure(figsize=(20, 6))
plt.hist(Toyota1['Weight'],facecolor ="yellow",edgecolor ="blue",bins =15)
plt.ylabel("Frequency");
plt.xlabel(" Total Weight")
plt.show()


# In[36]:


fuel_count = pd.value_counts(Toyota1['cc'].values, sort = True)
plt.xlabel('Frequency')
plt.ylabel('cc')
plt.title('Bar plot of cc')
fuel_count.plot.barh()


# In[37]:


sns.set(style = 'darkgrid')
sns.regplot(x = Toyota1['Age'], y = Toyota1['Price'], marker = '*')


# In[38]:


#Build individual model


# In[39]:


correlation_values= Toyota1.corr()
correlation_values


# In[40]:


import statsmodels.formula.api as smf


# In[41]:


m1= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= Toyota1).fit()


# In[42]:


m1.summary()


# In[43]:


m1_cc = smf.ols("Price~cc",data= Toyota1).fit()


# In[44]:


m1_cc.summary()


# In[45]:


m1_doors = smf.ols("Price~Doors", data= Toyota1).fit()


# In[46]:


m1_doors.summary()


# In[47]:


m1_to = smf.ols("Price~cc+Doors",data= Toyota1).fit()


# In[48]:


m1_to.summary()


# In[49]:


import statsmodels.api as sm


# In[50]:


sm.graphics.influence_plot(m1)


# In[51]:


Toyota2= Toyota1.drop(Toyota.index[[80]],axis=0)


# In[52]:


m2= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= Toyota2).fit()


# In[53]:


m2.summary()


# In[54]:


Toyota3 = Toyota1.drop(Toyota.index[[80,221]],axis=0)


# In[55]:


m3= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= Toyota3).fit()


# In[56]:


m3.summary()


# In[57]:


Toyota4= Toyota1.drop(Toyota.index[[80,221,960]],axis=0)


# In[58]:


m4= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = Toyota4).fit()


# In[59]:


m4.summary()


# In[60]:


#final model


# In[61]:


Finalmodel = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = Toyota4).fit()


# In[62]:


Finalmodel.summary()


# In[63]:


#predictions


# In[64]:


Finalmodel_pred = Finalmodel.predict(Toyota4)
Finalmodel_pred


# In[65]:


#validation


# In[66]:


plt.scatter(Toyota4["Price"],Finalmodel_pred,c='r');plt.xlabel("Observed values");plt.ylabel("Predicted values")


# In[67]:


#residual values vs fitted values


# In[68]:


plt.scatter(Finalmodel_pred, Finalmodel.resid_pearson,c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")


# In[69]:


plt.hist(Finalmodel.resid_pearson) 


# In[70]:


#QQ p;ot


# In[72]:


import pylab
import scipy.stats as st


# In[73]:


st.probplot(Finalmodel.resid_pearson, dist='norm',plot=pylab)


# In[74]:


# Testing of Final Model


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


train_data,test_Data= train_test_split(Toyota1,test_size=0.3)


# In[77]:


Finalmodel1 = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = train_data).fit()


# In[78]:


Finalmodel1.summary()


# In[79]:


Finalmodel_pred = Finalmodel1.predict(train_data)
Finalmodel_pred


# In[80]:


# Training and Testing of Residual Data


# In[81]:


Finalmodel_res = train_data["Price"]-Finalmodel_pred
Finalmodel_res


# In[82]:


Finalmodel_rmse = np.sqrt(np.mean(Finalmodel_res*Finalmodel_res))
Finalmodel_rmse


# In[83]:


Finalmodel_testpred = Finalmodel1.predict(test_Data)
Finalmodel_testpred


# In[84]:


Finalmodel_testres= test_Data["Price"]-Finalmodel_testpred
Finalmodel_testres


# In[85]:


Finalmodel_testrmse = np.sqrt(np.mean(Finalmodel_testres*Finalmodel_testres))
Finalmodel_testrmse


# In[ ]:





# In[ ]:




