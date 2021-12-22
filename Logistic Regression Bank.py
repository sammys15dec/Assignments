#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sohail's Logistic Regression Assignment


# In[2]:


#import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,accuracy_score,confusion_matrix,recall_score,precision_score,f1_score, auc
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


# In[3]:


#load data
bank = pd.read_csv("bank-full.csv",sep=";")
bank.head(10)


# In[4]:


#data analysis


# In[5]:


bank.info()


# In[6]:


bank.shape


# In[7]:


bank.dtypes


# In[8]:


bank.describe()


# In[9]:


bank.corr()


# In[10]:


#Graphical Visualization


# In[11]:


g = sns.pairplot(bank)
g.set(xticklabels=[])
plt.show()


# In[12]:


plt.style.use('seaborn-whitegrid')
bank.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# In[16]:


f, ax = plt.subplots(1,2, figsize=(16,8))

colors = ["#64FE2E", "#FF0000"]
labels ="Did not Open Term Suscriptions", "Opened Term Suscriptions"

plt.suptitle('Information on Term Suscriptions', fontsize=20)

bank["y"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=25)


# ax[0].set_title('State of Loan', fontsize=16)
ax[0].set_ylabel('% of Condition of Loans', fontsize=14)

# sns.countplot('loan_condition', data=bank, ax=ax[1], palette=colors)
# ax[1].set_title('Condition of Loans', fontsize=20)
# ax[1].set_xticklabels(['Good', 'Bad'], rotation='horizontal')
palette = ["#64FE2E", "#FA5858"]

sns.barplot(x="education", y="balance", hue="y", data=bank, palette=palette, estimator=lambda x: len(x) / len(bank) * 100)
ax[1].set(ylabel="(%)")
ax[1].set_xticklabels(bank["education"].unique(), rotation=0, rotation_mode="anchor")
plt.show()


# In[17]:


plt.style.use('seaborn-whitegrid')
bank.hist(bins=20, figsize=(15,10), color='red')
plt.show() 


# In[18]:


#label encoding


# In[19]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
bank["job"] = label_encoder.fit_transform(bank["job"])
bank["marital"] = label_encoder.fit_transform(bank["marital"])
bank["education"] = label_encoder.fit_transform(bank["education"])  
bank["default"] = label_encoder.fit_transform(bank["default"])
bank["housing"] = label_encoder.fit_transform(bank["housing"]) 
bank["loan"] = label_encoder.fit_transform(bank["loan"])
bank["contact"] = label_encoder.fit_transform(bank["contact"])
bank["month"] = label_encoder.fit_transform(bank["month"])
bank["poutcome"] = label_encoder.fit_transform(bank["poutcome"])
bank["y"] = label_encoder.fit_transform(bank["y"])


# In[20]:


bank.corr()['y'][:].plot.bar()


# In[21]:


import numpy as np
fig, ax = plt.subplots(figsize=(13,10))

mask = np.zeros_like(bank.corr())
mask[np.triu_indices_from(mask, 1)] = True

sns.heatmap(bank.corr(), annot=True,mask=mask, cmap='viridis',linewidths=0.5,ax=ax, fmt='.3f')

rotx = ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
roty = ax.set_yticklabels(ax.get_yticklabels(), rotation=30)


# In[22]:


duration_campaign = sns.scatterplot(x='duration', y='campaign',data = bank,
                     hue = 'y')

plt.axis([0,65,0,65])
plt.ylabel('Number of Calls')
plt.xlabel('Duration of Calls (Minutes)')
plt.title('The Relationship between the Number and Duration of Calls')
plt.show()


# In[23]:


# dropping the case number columns as it is not required
bank1= bank.iloc[:,[0,1,2,3,4,5,6,7,16]]


# In[24]:


bank1


# In[25]:


bank1.isna().sum()


# In[26]:


# Dividing our data into input and output variables 
X = bank1.iloc[:,:-2]
X


# In[27]:


Y = bank1.iloc[:,-1]
Y


# In[28]:


#build model


# In[29]:


#split into training and test data


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.3, random_state=0)


# In[31]:


print(X_train.shape)


# In[32]:


print(X_test.shape)


# In[34]:


from sklearn.linear_model import LogisticRegression
lr_clf=LogisticRegression()
lr_clf.fit(X_train, y_train)


# In[35]:


print("Training Accuracy :\t ", lr_clf.score(X_train, y_train))


# In[36]:


print("Testing Accuracy :\t  ",  lr_clf.score(X_test, y_test))


# In[37]:


y_pred = lr_clf.predict(X_test)
y_pred


# In[38]:


from sklearn import metrics
metrics.confusion_matrix(y_pred, y_test)


# In[39]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print (cm)


# In[40]:


#confusion matrix


# In[41]:


cm = confusion_matrix(y_test, y_pred)

class_label = ["Positive", "Negative"]
cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[42]:


print(classification_report(y_test, y_pred))


# In[43]:


#ROC Curve


# In[46]:


y_predictProb = lr_clf.predict_proba(X_train)

fpr, tpr, thresholds = roc_curve(y_train, y_predictProb[::,1])

roc_auc = auc(fpr, tpr)

print("auc :-",roc_auc)


# In[54]:


plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# In[55]:


prob=lr_clf.predict_proba(X_train)
prob=prob[:,1]

new_pred= pd.DataFrame({'actual': y_test,"pred":0})


# In[56]:


new_pred


# In[57]:


cm_new=confusion_matrix(new_pred.actual,new_pred.pred)
cm_new


# In[58]:


print(classification_report(new_pred.actual,new_pred.pred))


# In[59]:


#done

