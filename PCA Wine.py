#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohails's PCA Assignment


# In[3]:


#impoimport pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale


# In[4]:


#load data
wine = pd.read_csv("wine.csv")
wine.head(10)


# In[5]:


#analyze data


# In[6]:


wine.shape


# In[7]:


wine.dtypes


# In[8]:


wine.info()


# In[9]:


wine.describe()


# In[11]:


wine.data = wine.iloc[:,1:]


# In[12]:


wine.data.head()


# In[13]:


wine_normal = scale(wine.data)


# In[14]:


wine_normal


# In[15]:


pca = PCA()
pca_values = pca.fit_transform(wine_normal)


# In[16]:


pca_values


# In[17]:


pca = PCA(n_components = 6)
pca.fit(wine)


# In[19]:


#pca_values = pca.fit_transform(uni_normal)


# In[20]:


# Graphical Visualization


# In[21]:


sns.pairplot(pd.DataFrame(pca_values))


# In[22]:


# PCA


# In[23]:


var = pca.explained_variance_ratio_.cumsum()
var


# In[24]:


# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[25]:


pca.components_


# In[26]:


plt.plot(var1,color="red")


# In[27]:


x = pca_values[:,0:1]
y = pca_values[:,1:2]
#z = pca_values[:2:3]
plt.scatter(x,y)


# In[28]:


finalDf = pd.concat([pd.DataFrame(pca_values[:,0:7],columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7']), wine[['Type']]], axis = 1)


# In[29]:


finalDf


# In[30]:


sns.scatterplot(data=finalDf,x='pc1',y='pc2',hue='Type')


# In[31]:


pcavalues=pd.DataFrame(pca_values[:,:7],columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7'])


# In[32]:


pcavalues


# In[33]:


# Hierarichal Clustering


# In[34]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(pcavalues, method='complete'))


# In[35]:


# No infrences can be derived from the dendrogram.
# Go for Kmean Clustering for large data sets.


# In[36]:


# Kmeans


# In[37]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[38]:


k = list(range(2,8))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(pcavalues)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(pcavalues.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,pcavalues.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
TWSS


# In[39]:


#Elbow Chart
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[40]:


kmeans_clust=KMeans(n_clusters=5)
kmeans_clust.fit(pcavalues)
Clusters=pd.DataFrame(kmeans_clust.labels_,columns=['Clusters'])
Clusters


# In[41]:


wine['h_clusterid'] = pd.DataFrame(Clusters)


# In[42]:


wine


# In[43]:


# Grouping data for further predictions


# In[44]:


result=wine.iloc[:,1:].groupby(wine.h_clusterid).mean()


# In[45]:


result


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




