#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Association Rules Assignment


# In[3]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[4]:


#load data
movies = pd.read_csv("my_movies.csv")
movies


# In[5]:


# Pre-Processing


# In[6]:


df=pd.get_dummies(movies)
df


# In[7]:


df.describe()


# In[8]:


# Apriori Algorithm


# In[9]:


frequent_itemsets = apriori(df,min_support=0.5,use_colnames=True)
frequent_itemsets


# In[10]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[11]:


len(rules)


# In[19]:


rules_1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
rules_1


# In[13]:


rules.sort_values('lift',ascending = False)


# In[14]:


rules[rules.lift>1]


# In[15]:


# Graphical Visualization


# In[16]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[17]:


plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()


# In[18]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))


# In[ ]:





# In[ ]:




