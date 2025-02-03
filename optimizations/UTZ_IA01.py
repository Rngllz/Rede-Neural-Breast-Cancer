import pandas as pd
import tensorflow as tf
import sklearn
import scikeras
import numpy as np

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

k.clear_session()
rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape = (30,)),
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
        tf.keras.layers.Dropout(rate = 0.2),
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
        tf.keras.layers.Dropout(rate = 0.2),
        tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)
rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])


rede_neural.fit(X, y, patch_size = 10, epochs = 100)

novo = np.array([[ 
                15.32, 23.12, 2.1, 8.23, 0.223,
                15.32, 23.12, 2.1, 8.23, 0.223,
                15.32, 23.12, 2.1, 8.23, 0.223,  # 1 linha tem 30 características (colunas)                
                15.32, 23.12, 2.1, 8.23, 0.223,
                15.32, 23.12, 2.1, 8.23, 0.223,
                15.32, 23.12, 2.1, 8.23, 0.223
                 ]])

previsao = rede_neural.predict(novo)             # Trás o resultado da previsao            
print(previsao)