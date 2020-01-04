#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[2]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[3]:


data = pd.read_csv("C:/Users/admin/Videos/london_merged.csv",parse_dates=[0], index_col=[0])


# In[34]:


data.tail(3)


# In[6]:


data = data.resample('D').sum()


# In[7]:


df = data[['cnt']]


# In[8]:


df.head(3)


# In[24]:


data_training = df[:584]['cnt']
data_test = df[584:]['cnt']


# In[25]:


x_train = []
y_train = []


# In[26]:


for i in range(4, len(data_training)-4):
    x_train.append(data_training[i-4:i])
    y_train.append(data_training[i:i+4])


# In[27]:


x_train = np.array(x_train)


# In[28]:


y_train = np.array(y_train)


# In[30]:


x_train.shape, y_train.shape


# In[32]:


x_scaler = MinMaxScaler()
x_train = x_scaler.fit_transform(x_train)
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)


# In[52]:


x_train = x_train.reshape(576,4,1)


# In[53]:


reg = Sequential()

reg.add(LSTM(units=200, return_sequences=True, input_shape=(4,1)))
reg.add(Dropout(0.2))

reg.add(LSTM(units=200, return_sequences=True, input_shape=(4,1)))
reg.add(Dropout(0.15))

reg.add(LSTM(units=176, return_sequences=True, input_shape=(4,1)))
reg.add(Dropout(0.10))

reg.add(Dense(1))


# In[54]:


reg.compile(loss='mse', optimizer='adam')


# In[55]:


reg.fit(x_train, y_train, epochs=200, batch_size=4)


# In[56]:


x_test = []
y_test = []


# In[57]:


for i in range(4, len(data_test)-4):
    x_test.append(data_test[i-4:i])
    y_test.append(data_test[i:i+4])


# In[58]:


x_test , y_test = np.array(x_test), np.array(y_test)


# In[59]:


x_test.shape, y_test.shape


# In[60]:


x_test = x_scaler.fit_transform(x_test)


# In[61]:


x_test = x_test.reshape(139,4,1)


# In[62]:


y_pred = reg.predict(x_test)


# In[63]:


y_pred.shape


# In[64]:


y_pred = y_pred.reshape(139,4)


# In[65]:


y_pred = y_scaler.transform(y_pred)


# In[66]:


from sklearn.metrics import mean_squared_error


# In[67]:


def evaluate_model(y_test,y_pred):
    scores=[]
    
    for i in range(y_test.shape[1]):
        mse = mean_squared_error(y_test[:,i],y_pred[:,i])
        rmse = np.sqrt(mse)
        scores.append(rmse)
        
    return scores


# In[69]:


evaluate_model(y_test,y_pred)


# In[73]:


np.std(y_test[3])


# # Linear Regression model has significantly less error has compared with ML (machine Learning) model. 
