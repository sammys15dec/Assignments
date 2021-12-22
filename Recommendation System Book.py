#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Recommendation System Assignment


# In[2]:


#import libraries
import pandas as pd
import numpy as np


# In[7]:


#load data
book_df = pd.read_csv('book (1).csv',encoding='latin-1')
book_df


# In[8]:


book_df1=book_df.drop(['Unnamed: 0'], axis=1).rename(columns={"User.ID": "User_ID", "Book.Title": "Book_Title","Book.Rating": "Book_Rating" })


# In[9]:


book_df1


# In[10]:


# Preprocessing


# In[11]:


#no. of unique users in the data
User_ID_unique=book_df1.User_ID.unique()


# In[12]:


User_ID_unique=pd.DataFrame(User_ID_unique)


# In[13]:


len(book_df1.Book_Title.unique())


# In[14]:


user_book_df = book_df1.pivot_table(index='User_ID',
                                 columns='Book_Title',
                                 values='Book_Rating')


# In[15]:


user_book_df


# In[16]:


# Build the Model


# In[17]:


user_book_df.index = book_df1.User_ID.unique()


# In[18]:


#Impute those NaNs with 0 values

user_book_df.fillna(0, inplace=True)


# In[19]:


user_book_df


# In[20]:


#Calculate Cosine Similarity between Users

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[39]:


user_similar = 1 - pairwise_distances( user_book_df.values,metric='cosine')


# In[40]:


user_similar


# In[41]:


user_similar.shape


# In[42]:


#Store the results in a dataframe
user_similar_df = pd.DataFrame(user_similar)


# In[43]:


#Set the index and column names to user ids 
user_similar_df.index = book_df1.User_ID.unique()
user_similar_df.columns = book_df1.User_ID.unique()


# In[44]:


user_similar_df.iloc[0:5, 0:5]


# In[46]:


np.fill_diagonal(user_similar, 0)
user_similar_df.iloc[0:5, 0:5]


# In[48]:


#Most Similar Users
user_similar_df.idxmax(axis=1)


# In[49]:


book_df1[(book_df1['User_ID']==276729) | (book_df1['User_ID']==276726)]


# In[50]:


user1=book_df1[book_df1['User_ID']==276729]


# In[51]:


user2=book_df1[book_df1['User_ID']==276726]


# In[52]:


user1.Book_Title


# In[53]:


user2.Book_Title


# In[54]:


pd.merge(user1,user2,on='Book_Title',how='outer')


# In[55]:


#done

