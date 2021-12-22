#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Hypothesis Testing


# In[2]:


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


# In[3]:


#load dataset
Customer = pd.read_csv('Costomer+OrderForm.csv')
Customer.head(10)


# In[4]:


#Analyzing Data


# In[5]:


Customer.shape


# In[6]:


Customer.dtypes


# In[7]:


Customer.info()


# In[8]:


Customer.describe()


# In[9]:


Phillippines_value=Customer['Phillippines'].value_counts()
print(Phillippines_value)


# In[10]:


Indonesia_value=Customer['Indonesia'].value_counts()
print(Indonesia_value)


# In[11]:


Malta_value=Customer['Malta'].value_counts()
print(Malta_value)


# In[12]:


India_value=Customer['India'].value_counts()
print(India_value)


# In[13]:


#Hypothesis Testing
chiStats = sp.stats.chi2_contingency([[271,267,269,280],[29,33,31,20]])


# In[14]:


print('Test t=%f p-value=%f' % (chiStats[0], chiStats[1]))


# In[15]:


print('Interpret by p-Value')


# In[16]:


if chiStats[1] < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[18]:


#critical value = 0.1
alpha = 0.05
critical_value = sp.stats.chi2.ppf(q = 1 - alpha,df=chiStats[2])
observed_chi_val = chiStats[0]


# In[19]:


print('Interpret by critical value')


# In[20]:


if observed_chi_val <= critical_value:
       print ('Null hypothesis cannot be rejected (variables are not related)')
else:
       print ('Null hypothesis cannot be excepted (variables are not independent)')


# In[ ]:


#Inference : Proportion of Percentage Defective across all center is same


# In[ ]:





# In[ ]:




