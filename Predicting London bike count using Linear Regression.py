#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[64]:


from sklearn.linear_model import LinearRegression


# In[85]:


from numpy import nan
import seaborn as sns
from sklearn.metrics import mean_squared_error


# In[10]:


data = pd.read_csv("C:/Users/admin/Videos/london_merged.csv", parse_dates=[0], index_col=[0])


# In[11]:


data.head(3)


# In[12]:


df_day = data.resample('D').sum()


# In[13]:


df_day.head(5)


# In[16]:


df_day['day'] = df_day.index.day 


# In[34]:


data = df_day.drop(['day'], axis=1)


# In[22]:


plt.figure(figsize=(12,6))
sns.lineplot(x=df_day.index, y='cnt', data=df_day)


# In[35]:


corrmat = data.corr()


# In[36]:


corrmat 


# In[37]:


sns.pairplot(corrmat)


# In[44]:


plt.figure(figsize=(16,12))
sns.heatmap(corrmat, linewidth=2, linecolor='blue', annot=True)


# ## identifying high correlation and multicollinearity problem 
# ## 1) weather code high negative correlation
# ##  2) t1 is highly correlated with t2

# In[90]:


data.head(3)


# In[47]:


data.shape


# In[48]:


x = np.array(data.drop(['cnt'],axis=1))


# In[50]:


pd.DataFrame(x)


# In[51]:


y = np.array(data.cnt)


# In[52]:


pd.DataFrame(y)


# In[53]:


from sklearn.model_selection import train_test_split


# In[79]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)


# In[80]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[81]:


lr = LinearRegression()
lr.fit(x_train, y_train)


# In[82]:


lr_confidence = lr.score(x_test, y_test)
print('confidence:', lr_confidence)


# In[83]:


model_prediction = lr.predict(x_test)


# In[86]:


model_error = mean_squared_error(model_prediction, y_test)


# In[87]:


np.sqrt(model_error)


# In[88]:


pd.DataFrame(model_prediction)


# In[91]:


test = pd.DataFrame(data=y_test, columns=['True Value'])
prediction = pd.DataFrame(data=model_prediction, columns=['LR'])


# In[92]:


test.head(3)


# In[93]:


LR_model = pd.concat([test, prediction], axis=1)


# In[94]:


LR_model.head(3)


# In[96]:


LR_model.to_excel("C:/Users/admin/Videos/LR_model.xlsx")


# In[ ]:




