import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import torch as tc
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Heart_Disease_Prediction.csv')


print(df.info())
print(df.head())


#normalize data for ranges from 0-1
#use label encoding for numerical data, use onehot encoding for object data

scaler = StandardScaler()
le = LabelEncoder()

num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

df[num_cols] = scaler.fit_transform(df[num_cols])
df['Heart Disease'] = le.fit_transform(df['Heart Disease'])

print(df.head())


#define features and target (x and y)
x = df.drop(columns = ["Heart Disease"])

y = df['Heart Disease']

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state = 42, stratify=y
)

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")

class HeartMLP(Model):
    def __init__(self):
        super(HeartMLP, self).__init__()
        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(8, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
        
model = HeartMLP()

model.compile(
    optimizer='adam',
    loss="binary_crossentropy",
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=70,
    batch_size=8
)


#test_loss, test_acc = model.evaluate(x_val, y_val)
#print("Test accuracy: ", test_acc)

