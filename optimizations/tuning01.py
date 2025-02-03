import pandas as pd
import tensorflow as tf
import sklearn
import scikeras

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k
from sklearn.model_selection import GridSearchCV

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

def criar_rede(optimizer, loss, kernel_initializer, activation, neurons): # Os valores fixos foram transformados
    k.clear_session()                                                     # em parâmetros
    rede_neural = Sequential([
            tf.keras.layers.InputLayer(shape = (30,)),
            tf.keras.layers.Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer),
            tf.keras.layers.Dropout(rate = 0.2),
            tf.keras.layers.Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer),
            tf.keras.layers.Dropout(rate = 0.2),
            tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
        ])
    rede_neural.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return rede_neural

rede_neural = KerasClassifier(model = criar_rede)

parametros = {                      
        'batch_size':  [10, 30],
        'epochs': [50, 100],
        'model__optimizer': ['Adam', 'sgd'],                         # Posso escolher quantos parâmetros eu quiser   
        'model__loss': ['binary_crossentropy', 'hinge'],             # Ao todo são 128 combinações de parâmetros
        'model__kernel_initializer': ['random_uniform','normal'],
        'model_activation': ['relu', 'tahn'],
        'model__neurons': [16, 8]
    }

grid_search = GridSearchCV(
        estimator = rede_neural,
        param_grid = parametros,
        scoring = 'accuracy',
        cv = 10                                                      # Quanto maior o CV mais demorado, multiplica as
    )                                                                # 128 combinações, dando 1280. Demoraria HORAS


grid_search = grid_search.fit(X,y)                                   # Executa o treinamento
melhores_parametros = grid_search.best_params_                       # Trás os melhores parâmetros a serem utilizados
melhor_precisao = grid_search.best_score_                            # Trás o melhor resultado