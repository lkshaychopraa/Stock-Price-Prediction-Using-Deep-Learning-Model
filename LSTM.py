#!/usr/bin/env python
# coding: utf-8

# In[276]:


import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error


# In[5]:


key = '74d877047a3fb120cb0dbdc0cd8fbd0f22b65cbe'


# In[6]:


df = pdr.get_data_tiingo('AAPL',api_key = key)


# In[7]:


df.to_csv('AAPL.csv')


# In[8]:


df = pd.read_csv('AAPL.csv')


# In[9]:


df2= df.reset_index()['close']


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df1 = df.reset_index()['close']


# In[13]:


df1.shape


# In[14]:


df1


# In[15]:


plt.plot(df1,color = 'green')
plt.xlabel('Day')
plt.ylabel('AAPL Price')
plt.title('AAPL Price 5 year data')


# <center><B>LSTM are sensitive to the scale of the data, so we apply MinMax scaler</B></center>

# In[16]:


scaler = MinMaxScaler(feature_range=(0,1))


# In[17]:


df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[102]:


print(df1)


# In[18]:


#Splitting the data into training data and test data 
#is crucial to determine efficiency of the model


# In[19]:


training_size = int(len(df1)*0.65)
test_size = len(df1)- training_size
train_data,test_data = df1[0:training_size,:], df1[training_size:len(df1),:1]


# In[105]:


training_size, test_size


# In[106]:


train_data


# In[20]:


#Convert an array of values into dataset matrix
def create_dataset(dataset,time_step=1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i +time_step,0])
    return np.array(dataX), np.array(dataY)


# In[21]:


#reshape into X = t, t+1, t+2, t+3 and Y= t+4

time_step = 100
X_train, y_train = create_dataset(train_data,time_step)
X_test,ytest = create_dataset(test_data,time_step)


# In[22]:


print(X_train)


# In[23]:


print(X_test.shape), print(ytest.shape)


# In[25]:


#reshape input to be [samples,time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[107]:


model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape = (100,1)))
model.add(LSTM(50, return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[108]:


model.summary()


# In[109]:


model.fit(X_train,y_train,validation_data = (X_test,ytest),epochs = 100,batch_size = 64,verbose = 1)


# In[111]:


#Prediction and performance metrics
import tensorflow as tf
tf.__version__


# In[44]:


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[45]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[68]:


math.sqrt(mean_squared_error(y_train,train_predict))


# In[69]:


####Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[67]:


###plotting
#shift train prediction for plotting
look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:,:]= np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict

#shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(train_predict) +look_back*2 +1:len(df1)-1,:]=test_predict

#plot baseline and predictions
plt.plot(scaler.inverse_transform(df1),label = 'overall')
plt.plot(trainPredictPlot,label = 'training data')
plt.plot(testPredictPlot, label = 'test data')
plt.xlabel('Days',color = 'red')
plt.ylabel('AAPL Stock Price', color = 'red')
plt.legend()
plt.show


# In[112]:


len(test_data)


# In[267]:


x_input = test_data[340:].reshape(1,-1)
x_input.shape


# In[268]:


temp_input = list(x_input)
temp_input = temp_input[0].tolist()
temp_input


# In[269]:


from numpy import array


# In[270]:


lst_output =[]
n_steps = 100
i = 0
while(i<30):
    if(len(temp_input)>100):
        #print(temp_input)
        x_input = np.array(temp_input[1:])
        print('{} day input{}'.format(i,x_input))
        x_input = x_input.reshape(-1,1)
        x_input = x_input.reshape((1,n_steps,1))
        #Print(x_input)
        yhat = model.predict(x_input,verbose = 0)
        print('{} day output{}'.format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i +1
    else:
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input,verbose = 0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i +1
print(lst_output)
len(lst_output)


# In[271]:


len(df1)


# In[272]:


df3 = df1.tolist()
df3.extend(lst_output)


# In[273]:


day_new = np.arange(1,101)
day_pred = np.arange(101,131)
len(day_pred)


# In[274]:


plt.plot(day_new,scaler.inverse_transform(df1[1157:]), label = 'test data ')
plt.plot(day_pred,scaler.inverse_transform(lst_output),label = '60 day Prediction')
plt.legend()
plt.xlabel('days')
plt.ylabel('AAPL Stock Prices')
plt.title('LSTM MODEL FOR AAPL STOCK')


# In[275]:


df3 = df1.tolist()
df3.extend(lst_output)
df3 = scaler.inverse_transform(df3).tolist()
plt.plot(df3)


# In[ ]:




