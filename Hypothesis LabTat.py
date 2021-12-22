#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Hypothesis Testing


# In[2]:


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


# In[4]:


#load dataset
LabTAT =pd.read_csv('LabTAT.csv')
LabTAT.head(10)


# In[6]:


#data analysis


# In[7]:


LabTAT.shape


# In[8]:


LabTAT.dtypes


# In[9]:


LabTAT.info()


# In[10]:


LabTAT.describe()


# In[11]:


Laboratory_1=LabTAT['Laboratory 1'].mean()
print('Laboratory 1 Mean = ',Laboratory_1)


# In[14]:


Laboratory_2=LabTAT['Laboratory 2'].mean()
print('Laboratory 2 Mean = ',Laboratory_2)


# In[12]:


Laboratory_3=LabTAT['Laboratory 3'].mean()
print('Laboratory 3 Mean = ',Laboratory_3)


# In[13]:


Laboratory_4=LabTAT['Laboratory 4'].mean()
print('Laboratory 4 Mean = ',Laboratory_4)


# In[15]:


print('Laboratory_1 > Laboratory_2 = ',Laboratory_1 > Laboratory_2)
print('Laboratory_2 > Laboratory_3 = ',Laboratory_2 > Laboratory_3)
print('Laboratory_3 > Laboratory_4 = ',Laboratory_3 > Laboratory_4)
print('Laboratory_4 > Laboratory_1 = ',Laboratory_4 > Laboratory_1)


# In[18]:


#The Null and Alternative Hypothesis

There are no significant differences between the groups' mean Lab values. H0:μ1=μ2=μ3=μ4

There is a significant difference between the groups' mean Lab values. Ha:μ1≠μ2≠μ3≠μ4


# In[19]:


# Visualization


# In[20]:


sns.distplot(LabTAT['Laboratory 1'])


# In[21]:


sns.distplot(LabTAT['Laboratory 2'])


# In[22]:


sns.distplot(LabTAT['Laboratory 3'])


# In[23]:


sns.distplot(LabTAT['Laboratory 4'])


# In[24]:


sns.distplot(LabTAT['Laboratory 1'])
sns.distplot(LabTAT['Laboratory 2'])
sns.distplot(LabTAT['Laboratory 3'])
sns.distplot(LabTAT['Laboratory 4'])
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# In[25]:


sns.boxplot(data=[LabTAT['Laboratory 1'],LabTAT['Laboratory 2'],LabTAT['Laboratory 3'],LabTAT['Laboratory 4']],notch=True)
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# In[26]:


#Hypithesis Testing


# In[27]:


alpha=0.05
Lab1=pd.DataFrame(LabTAT['Laboratory 1'])
Lab1


# In[28]:


Lab2=pd.DataFrame(LabTAT['Laboratory 2'])
Lab2


# In[29]:


Lab3=pd.DataFrame(LabTAT['Laboratory 3'])
Lab3


# In[33]:


Lab4=pd.DataFrame(LabTAT['Laboratory 4'])
Lab4


# In[34]:


print(Lab1,Lab2,Lab3,Lab4)


# In[35]:


tStat, pvalue = sp.stats.f_oneway(Lab1,Lab2,Lab3,Lab4)


# In[36]:


print("P-Value:{0} T-Statistic:{1}".format(pvalue,tStat))


# In[37]:


if pvalue < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[ ]:


#Inference : There is no significant difference in the average TAT for all the labs.


# In[ ]:





# In[ ]:





# In[ ]:




