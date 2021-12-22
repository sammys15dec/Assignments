#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Hypothesis Testing


# In[3]:


#import libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest


# In[11]:


#load dataset
BuyerRatio =pd.read_csv('BuyerRatio.csv')
BuyerRatio.head(10)


# In[13]:


#Data Analysis


# In[6]:


BuyerRatio.shape


# In[7]:


BuyerRatio.dtypes


# In[8]:


BuyerRatio.info


# In[9]:


BuyerRatio.describe()


# In[10]:


East=BuyerRatio['East'].mean()
print('East Mean = ',East)


# In[11]:


West=BuyerRatio['West'].mean()
print('West Mean = ',West)


# In[12]:


North=BuyerRatio['North'].mean()
print('North Mean = ',North)


# In[13]:


South=BuyerRatio['South'].mean()
print('South Mean = ',South)


# In[ ]:


#Null and Alternate Hypothesis
No significant difference between groups 'mean value. H0:μ1=μ2=μ3=μ4'

there is a significant difference between the groups' mean values. Ha:μ1≠μ2≠μ3≠μ4'


# In[14]:


#visualization
sns.distplot(BuyerRatio['East'])
sns.distplot(BuyerRatio['West'])
sns.distplot(BuyerRatio['North'])
sns.distplot(BuyerRatio['North'])


# In[15]:


sns.distplot(BuyerRatio['East'])


# In[16]:


sns.distplot(BuyerRatio['West'])


# In[17]:


sns.distplot(BuyerRatio['North'])


# In[18]:


sns.distplot(BuyerRatio['South'])


# In[19]:


sns.distplot(BuyerRatio['East'])
sns.distplot(BuyerRatio['West'])
sns.distplot(BuyerRatio['North'])
sns.distplot(BuyerRatio['South'])
plt.legend(['East','West','North','South'])


# In[20]:


sns.boxplot(data=[BuyerRatio['East'],BuyerRatio['West'],BuyerRatio['North'],BuyerRatio['South']],notch=True)
plt.legend(['East','West','North','South'])


# In[22]:



alpha=0.05
Male = [50,142,131,70]
Female=[435,1523,1356,750]
Sales=[Male,Female]
print(Sales)


# In[23]:


chiStats = sp.stats.chi2_contingency(Sales)
print('Test t=%f p-value=%f' % (chiStats[0], chiStats[1]))
print('Interpret by p-Value')


# In[24]:


if chiStats[1] < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[25]:


alpha = 0.05
critical_value = sp.stats.chi2.ppf(q = 1 - alpha,df=chiStats[2])
critical_value 


# In[26]:


observed_chi_val = chiStats[0]
print('Interpret by critical value')


# In[27]:


if observed_chi_val <= critical_value:
    print ('Null hypothesis cannot be rejected (variables are not related)')
else:
    print ('Null hypothesis cannot be excepted (variables are not independent)')


# In[14]:


#Inference : Proportion of male and female across regions is same

