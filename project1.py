import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv1D,MaxPool1D,BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam
dataset = pd.read_csv('D:\progaming languages\python\SantanderBank.csv')
#print(dataset.shape)
print("Preview of data\n")
print(dataset.head())
print("\n---------------------------------------------------------\n")
x=dataset.drop(['ID','TARGET'], axis=1)
y=dataset['TARGET']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)
print("Total data\n")
print(x_train.shape,x_test.shape)
filter=VarianceThreshold(0.2)
x_train=filter.fit_transform(x_train)
x_test=filter.transform(x_test)
print("Removing dulicates and remaning data\n")
print(x_train.shape,end="")
print(x_test.shape)
x_train_T=x_train.T
x_test_T=x_test.T
x_train_T=pd.DataFrame(x_train_T)
x_test_T=pd.DataFrame(x_test_T)
print("No.of Duplicates in dataset")
print(x_train_T.duplicated().sum())
duplicated=x_train_T.duplicated()
features_to_keep=[not index for index in duplicated]
x_train=x_train_T[features_to_keep].T
print("No.of non-Duplicate dataset")
print(x_train.shape,end="")
x_test=x_test_T[features_to_keep].T
print(x_test.shape)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
model=Sequential()
model.add(Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(254,1)))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
index=19
print("Total summary")
print(model.summary())
model.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
history={'accuracy': [0.9549460411071777,
  0.9584648609161377,
  0.959484338760376,
  0.9600762724876404,
  0.9601420760154724,
  0.9601585268974304,
  0.9601913690567017,
  0.9601420760154724,
  0.9602736234664917,
  0.9602900743484497,
  0.960322916507721,
  0.9602407217025757,
  0.9602900743484497,
  0.960322916507721,
  0.9602900743484497,
  0.9601913690567017,
  0.960339367389679,
  0.9602736234664917,
  0.960355818271637,
  0.960322916507721],
 'loss': [0.20062212646007538,
  0.17323994636535645,
  0.16500122845172882,
  0.162405326962471,
  0.16038189828395844,
  0.15759964287281036,
  0.1557125449180603,
  0.15397605299949646,
  0.15129274129867554,
  0.15005743503570557,
  0.14902569353580475,
  0.1486220508813858,
  0.14752936363220215,
  0.14704017341136932,
  0.14653046429157257,
  0.14576765894889832,
  0.1452939659357071,
  0.14524489641189575,
  0.14516344666481018,
  0.14479608833789825],
 'val_accuracy': [0.9596816897392273,
  0.960339367389679,
  0.9606024622917175,
  0.9605367183685303,
  0.960405170917511,
  0.9604709148406982,
  0.9604709148406982,
  0.9605367183685303,
  0.960405170917511,
  0.960405170917511,
  0.960339367389679,
  0.9605367183685303,
  0.9606024622917175,
  0.960339367389679,
  0.9602736234664917,
  0.9602736234664917,
  0.9600762724876404,
  0.9602736234664917,
  0.960405170917511,
  0.960339367389679],
 'val_loss': [0.17072658240795135,
  0.17098182439804077,
  0.16867370903491974,
  0.1678592413663864,
  0.1625373810529709,
  0.16285011172294617,
  0.15724265575408936,
  0.15354587137699127,
  0.1528950184583664,
  0.14891521632671356,
  0.14836306869983673,
  0.1495659202337265,
  0.14729507267475128,
  0.14798125624656677,
  0.14874811470508575,
  0.14703427255153656,
  0.14765673875808716,
  0.14697785675525665,
  0.1460145115852356,
  0.1464858204126358]}
history['val_accuracy'][index]=0.1464858204126358
print('total customers satisfied=',history['accuracy'][index])
print('Total customers unsatisfied=',history['val_accuracy'][index])
