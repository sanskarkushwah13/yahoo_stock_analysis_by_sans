import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data  as pdr
import yfinance as yf
import streamlit as st
from keras.models import load_model

st.title("Stock Price Analysis")

start = '2014-01-01'
end = '2023-12-31'
yf.pdr_override()

user_input=st.text_input("enter stock ","AAPL")
df =pdr.get_data_yahoo(user_input, start, end)

st.subheader("Date 2014 - 2023")
df.head()
st.write(df.describe())


st.subheader("Closing Price vs Time chart")
fig =plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time chart 100 mean avg")
ma100 = df.Close.rolling(100).mean()
fig =plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
st.pyplot(fig)




data_training = pd.DataFrame(df['Close'][0 : 1761])
data_test=pd.DataFrame(df['Close'][1761 : int(len(df))])

print("training data shape :",data_training.shape)
print("testing data shape",data_test.shape)
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler(feature_range=(0,1))
data_training_array =Scaler.fit_transform(data_training)

x_train =[]
y_train =[]

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)


#load model 

model=load_model('keras_model.h5')

#testing part
# final_df =past_100_days.append(data_test, ignore_index=True)
past_100_days =data_training.tail(100)
final_df=pd.concat([past_100_days,data_test],ignore_index=True)
input_data= Scaler.fit_transform(final_df)
# final_df
# input_data

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)
y_prediction=model.predict(x_test)
v=Scaler.scale_
scale_factor=1/v[0]
y_prediction=y_prediction*scale_factor
y_test=y_test*scale_factor

st.subheader("Orignal vs Pridicted")

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Orignal Price')
plt.plot(y_prediction,'r',label="prediction")
plt.xlabel('Time')
plt.ylabel('Price')
# plt.legend()
st.pyplot(fig)