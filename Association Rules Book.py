#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Association Rules Assignment


# In[2]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[3]:


#load data
book=pd.read_csv('book.csv')
book.head(10)


# In[4]:


book.info()


# In[5]:


te=TransactionEncoder()


# In[6]:


te_ary=te.fit(book).transform(book)
te_ary


# In[7]:


# Graphical Visualization


# In[9]:


sns.distplot(book['ChildBks'])


# In[10]:


frequent_itemsets=apriori(book,min_support=0.05,use_colnames=True,max_len=3)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)


# In[11]:


sns.pairplot(book)


# In[12]:


sns.barplot(book['CookBks'], book['ChildBks'])


# In[13]:


sns.boxplot(book['ChildBks'], book['CookBks'], hue=book['Florence'])


# In[14]:


sns.lmplot(x='ChildBks', y='CookBks', data=book)


# In[15]:


sns.jointplot(book['ChildBks'],book['CookBks'], kind="kde")


# In[16]:


# Preprocessing


# In[17]:


df=pd.get_dummies(book)
df


# In[18]:


df.describe()


# In[19]:


# Apriori Algorithm


# In[20]:


frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets


# In[21]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
rules


# In[22]:


len(rules)


# In[23]:


rules_1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules_1


# In[24]:


rules.sort_values('lift',ascending = False)


# In[25]:


rules[rules.lift>1]


# In[ ]:





# In[ ]:





# In[ ]:




