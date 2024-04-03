# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Develop a Recurrent Neural Network (RNN) model to predict the stock prices of Google. The goal is to train the model using historical stock price data and then evaluate its performance on a separate test dataset.

Dataset: The dataset consists of two CSV files:

Trainset.csv: This file contains historical stock price data of Google, which will be used for training the RNN model. It includes features such as the opening price of the stock.
Testset.csv: This file contains additional historical stock price data of Google, which will be used for testing the trained RNN model. Similarly, it includes features such as the opening price of the stock.
The objective is to build a model that can effectively learn from the patterns in the training data to make accurate predictions on the test data.

## Design Steps

### Step 1: 

Read and preprocess training data, including scaling and sequence creation.

### Step 2:

Initialize a Sequential model and add SimpleRNN and Dense layers.

### Step 3: 

Compile the model with Adam optimizer and mean squared error loss.

### Step 4: 

Train the model on the prepared training data.

### Step 5: 

Preprocess test data, predict using the trained model, and visualize the results.



## Program
#### Name:SAILESHKUMAR A
#### Register Number:212222230126

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values

type(train_set)

train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape

length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(60,input_shape=(60,1)))
model.add(layers.Dense(1))
model.compile(optimizer='adam',loss='mse')

print("NAME : SAILESHKUMAR A  REG NO. : 212222230126")
model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Name: SAILESHKUMAR A          Register Number:212222230126")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

## Output

### True Stock Price, Predicted Stock Price vs time

!![Screenshot 2024-04-03 113140](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/113497410/ae2fa9ee-37b9-4dfc-94ff-0bf8fe0dd6fc)



### Mean Square Error
![Screenshot 2024-04-03 115002](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/113497410/0445dfcd-7817-48ad-82f4-5713944fa348)
![Screenshot 2024-04-03 115204](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/113497410/54c47679-e110-424a-b637-14e24dafc777)
![Screenshot 2024-04-03 115150](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/113497410/b45c1ec3-b431-4d17-8132-d2b35eec5808)
![Screenshot 2024-04-03 115137](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/113497410/02b6cd79-101c-4724-a0c7-5ec6208265e0)
![Screenshot 2024-04-03 115124](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/113497410/3a657478-84eb-471b-85f9-409da5f9234e)
![Screenshot 2024-04-03 115039](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/113497410/62956035-1ab3-40ce-9ec8-e9d48bb30813)
![Screenshot 2024-04-03 115026](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/113497410/f411a4b0-339a-411c-98ac-4c32bd4ebc2a)
![Screenshot 2024-04-03 115012](https://github.com/etjabajasphin/rnn-stock-price-prediction/assets/113497410/89a5d1c1-8a89-415c-a882-f7d132ad39b9)





## Result

Thus a Recurrent Neural Network model for stock price prediction is done.
