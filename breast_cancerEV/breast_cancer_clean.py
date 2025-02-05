#-------------- Importação de Bibliotecas

import pandas as pd

import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k
from tensorflow.keras.layers import InputLayer, Dense, Dropout

import scikeras
from scikeras.wrappers import KerasClassifier


X = pd.read_csv('C:/Users/User/Documents/GitHub/Rede-Neural-Breast-Cancer/base_de_dados/entradas_breast.csv')
y = pd.read_csv('C:/Users/User/Documents/GitHub/Rede-Neural-Breast-Cancer/base_de_dados/saidas_breast.csv')

                    
#-------------- Modelando a Rede Neural
def criar_rede():
    # Modelando a rede
    k.clear_session()
    rede_neural = Sequential([
            InputLayer(shape = (30,)),
            Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
            Dropout(rate = 0.2),
            Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
            Dropout(rate = 0.2),
            Dense(units = 1, activation = 'sigmoid')
        ])
    rede_neural.summary() 
    
    # Configurando o treinamento
    otimizador = tf.keras.optimizers.Adam(
                                        learning_rate = 0.001, 
                                        clipvalue = 0.5        
                                        )
    rede_neural.compile(
        optimizer = otimizador,            
        loss = 'binary_crossentropy',  
        metrics = ['binary_accuracy'] 
        )
    return rede_neural


#-------------- Treinamento
rede_neural = KerasClassifier(
                model = criar_rede,
                epochs = 100,
                batch_size = 10
                )                                 


#--------------- Teste (Com validação Cruzada)
resultados = cross_val_score(
    estimator = rede_neural,
    X = X,
    y = y,
    cv = 10,
    scoring = 'accuracy'
    )

#--------------- Resultado
resultados.mean()
resultados.std() 
