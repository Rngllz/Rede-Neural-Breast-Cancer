import pandas as pd
import tensorflow as tf
import sklearn
import scikeras

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

def criar_rede():
    k.clear_session()
    rede_neural = Sequential([
            tf.keras.layers.InputLayer(shape = (30,)),
            tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
            tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
            tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
        ])
    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)
    rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return rede_neural

rede_neural = KerasClassifier(model = criar_rede, epochs = 100, batch_size = 10)

            # validação cruzada    
resultados = cross_val_score(
                 estimator = rede_neural, # rede neural já configurada
                 X = X,                   # Previsores
                 y = y,                   # Resultados
                 cv = 10,                 # Por quantas partes a base de dados será dividida e cruzada
                 scoring = 'accuracy'     # Avaliação
            )

resultados.mean() # Média dos 10 resultados de accuracy

resultados.std()   # Desvio padrão, indica a média de quanto cada resultado
                  # sozinho está longe da média final