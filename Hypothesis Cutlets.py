#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Sohail's Hypothesis Testing


# In[1]:


#Import libraries
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


# In[3]:


#load dataset
data1 = pd.read_csv("Cutlets.csv")
data1.head(10)


# In[8]:


#data analysis


# In[4]:


data1.shape


# In[5]:


data1.dtypes


# In[6]:


data1.info()


# In[7]:


data1.describe(include='all')


# In[9]:


Unit_A=data1['Unit A'].mean()


# In[10]:


Unit_B=data1['Unit B'].mean()


# In[11]:


print('Unit A Mean is ',Unit_A, '\nUnit B Mean is ',Unit_B)


# In[12]:


print('Unit A Mean > Unit B Mean = ',Unit_A>Unit_B)


# In[13]:


#visualization


# In[14]:


sns.distplot(data1['Unit A'])


# In[15]:


sns.distplot(data1['Unit B'])


# In[16]:


sns.distplot(data1['Unit A'])
sns.distplot(data1['Unit B'])
plt.legend(['Unit A','Unit B'])


# In[17]:


sns.boxplot(data=[data1['Unit A'],data1['Unit B']],notch=True)
plt.legend(['Unit A','Unit B'])


# In[18]:


alpha=0.05
UnitA=pd.DataFrame(data1['Unit A'])
UnitA


# In[19]:


UnitB=pd.DataFrame(data1['Unit B'])
UnitB


# In[20]:


print(UnitA,UnitB)


# In[21]:


tStat,pValue =sp.stats.ttest_ind(UnitA,UnitB)


# In[22]:


print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat))


# In[23]:


if pValue <0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[ ]:


#Inference : No significant difference in diameter of Unit A and Unit B


# In[ ]:




