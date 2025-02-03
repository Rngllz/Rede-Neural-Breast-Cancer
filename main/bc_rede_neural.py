import panda as pd
import tensorflow as tf
import numpy as np

import scikeras
from scikeras.wrappers import KerasClassifier

import sklearn
from sklern.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras import backend as k


X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')


def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):
    k.clear_session()
    rede_neural = Sequential([
            InputLayer(shape = (30,)),
            Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer),
            Dropout(rate = 0.2),
            Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer),
            Dropout(rate = 0.2),
            Dense(units = 1, activation = 'sigmoid')
        ])
    rede_neural.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return rede_neural

rede_neural = KerasClassifier(model = criar_rede)

parametros = {
        'batch_size': [10, 30],
        'pochs': [50, 100],
        'model__optimizer': ['Adam', 'sgd'],
        'model__loss': ['binary_crossentropy', 'hinge'],
        'model__kernel_initializer': ['random_uniform', 'normal'],
        'model__activation': ['relu', 'tahn'],
        'model__neurons': [16, 8]
    }

grid_search = GridSearchCV(
        estimator = rede_neural,
        param_grid = parametros,
        scoring = 'accuracy',
        cv = 10
    )

grid_search = grid_search.fit(X,y)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

novo = np.array([[ 
                15.32, 23.12, 2.1, 8.23, 0.223,
                15.32, 23.12, 2.1, 8.23, 0.223,
                15.32, 23.12, 2.1, 8.23, 0.223,
                15.32, 23.12, 2.1, 8.23, 0.223,
                15.32, 23.12, 2.1, 8.23, 0.223,
                15.32, 23.12, 2.1, 8.23, 0.223
                 ]])

previsao = grid_search.predict(novo)
print(previsao)



