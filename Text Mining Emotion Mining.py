#!/usr/bin/env python
# coding: utf-8

# In[35]:


# Sohail's Text Mining Assignment


# In[36]:


# import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[9]:


url = 'https://www.amazon.in/product-reviews/B089MT34QG/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=1'
reviewlist = []


# In[10]:


def get_soup(url):
    r = requests.get(url)
    print(r.text)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


# In[11]:


def get_reviews(soup):
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try:
        for item in reviews:
            review = {
            'product': soup.title.text.replace('Amazon.in:Customer reviews:', '').strip(),
            'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
            'rating':  float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
            'body': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            }
            reviewlist.append(review)
    except:
        pass


# In[21]:


for x in range(1,10):
    soup = get_soup(f'https://www.amazon.in/product-reviews/B089MT34QG/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber={x}')
    print(f'Getting page: {x}')
    get_reviews(soup)
    print(len(reviewlist))
    if not soup.find('li', {'class': 'a-last'}):
        pass
    else:
        break


# In[20]:


df = pd.DataFrame(reviewlist)
df.to_excel('rating.xlsx', index=False)
print('Fin.')


# In[16]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
import seaborn as sns
from textblob import TextBlob
from nltk import tokenize
from nltk.sentiment.util import *


# In[22]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


# In[27]:


data = pd.read_csv("rating.csv",encoding = "ISO-8859-1")
data.head()


# In[28]:


data.shape


# In[29]:


data.dtypes


# In[30]:


data.info()


# In[31]:


sns.distplot(data['rating'])


# In[32]:


sns.countplot(x='rating',data=data)


# In[33]:


df['rating'].value_counts


# In[34]:


df['rating'].isnull().sum()


# In[17]:


sid.polarity_scores(df.loc[0]['body'])


# In[18]:


df['scores'] = df['body'].apply(lambda body: sid.polarity_scores(body))
df.head()


# In[19]:


df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['sentiment_type']=''
df.loc[df.compound>0,'sentiment_type']='POSITIVE'
df.loc[df.compound==0,'sentiment_type']='NEUTRAL'
df.loc[df.compound<0,'sentiment_type']='NEGATIVE'


# In[20]:


df.head()


# In[21]:


df.sentiment_type.value_counts().plot(kind='bar',title="sentiment analysis")


# In[62]:


new_data = data.rename(columns = {"analysis": "Emotion"})
new_data.head(10)


# In[63]:


new_data['Emotion'].value_counts()


# In[64]:


new_data['Emotion'].value_counts().plot(kind='bar')


# In[67]:


import matplotlib.pyplot as plt
sns.countplot(new_data['Emotion']) #old 
plt.figure(figsize=(20,10)) #new 
sns.countplot(x='Emotion', data=new_data)
plt.show()


# In[69]:


new_data1=new_data[['product','Emotion','body']].copy()


# In[70]:


new_data1.Emotion.value_counts()


# In[ ]:




