#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sohail's Forecasting Assignment


# In[2]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# load data
Coca=pd.read_excel("CocaCola_Sales_Rawdata.xlsx")
Coca.head(20)


# In[4]:


Coca.shape


# In[5]:


Coca.dtypes


# In[6]:


Coca.info()


# In[7]:


Coca.describe()


# In[8]:


# Graphical Visualization


# In[9]:


sns.boxplot("Sales",data=Coca)


# In[10]:


sns.catplot("Quarter","Sales",data=Coca,kind="box")


# In[11]:


Coca.Sales.plot(label="org")
for i in range(2,10,2):
    Coca["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[12]:


Coca.Sales.plot()


# In[13]:


plt.figure(figsize=(28,6))
Coca.Sales.plot()


# In[14]:


# Preprocessing


# In[15]:


quarter=['Q1','Q2','Q3','Q4']
n=Coca['Quarter'][0]
n[0:2]


# In[16]:


Coca['quarter']=0
for i in range(42):
    n=Coca['Quarter'][i]
    Coca['quarter'][i]=n[0:2]
    dummy=pd.DataFrame(pd.get_dummies(Coca['quarter']))
    coco=pd.concat((Coca,dummy),axis=1)
t= np.arange(1,43)
coco['t']=t
coco['t_square']=coco['t']*coco['t']
log_Sales=np.log(coco['Sales'])
coco['log_Sales']=log_Sales


# In[17]:


coco


# In[18]:


plt.figure(figsize=(12,3))
sns.lineplot(x="quarter",y="Sales",data=Coca)


# In[19]:


# Splitting Data into training data and test data


# In[20]:


Train = coco.head(34)


# In[21]:


Test = coco.iloc[34:38,:]


# In[22]:


predict_data = coco.tail(4)


# In[23]:


Coca2= coco.iloc[0:38,:]


# In[24]:


Train


# In[25]:


Test


# In[26]:


predict_data


# In[27]:


# Build Model & Calculate RMSE Values


# In[28]:


# Linear Model
import statsmodels.formula.api as smf 
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[29]:


# Exponential
Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[30]:


# Quadratic 
Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# In[31]:


# Additive seasonality 
add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[32]:


# Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[33]:


# Multiplicative Seasonality
Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[34]:


# Multiplicative Additive Seasonality
Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea


# In[35]:


# Compare the results


# In[36]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[37]:


# Predict New Model


# In[38]:


predict_data


# In[39]:


# Build the model on entire data set


# In[40]:


model_full = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=Coca2).fit()


# In[41]:


pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
pred_new


# In[42]:


predict_data["forecasted_Sales"] = pd.DataFrame(pred_new)


# In[43]:


predict_data


# In[44]:


# done


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




