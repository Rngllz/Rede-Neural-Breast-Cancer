

#-------------- Importação de Bibliotecas

import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential


X = pd.read_csv('/content/entradas_breast.csv')
y = pd.read_csv('/content/saidas_breast.csv')

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.25)

X_treinamento.shape
y_treinamento.shape

X_teste.shape
y_teste.shape   
                    

#-------------- Modelando a Rede Neural

rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape = (30,)),
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
        tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])

rede_neural.summary() 

#-------------- Configurando a Rede Neural
                                                        
otimizador = tf.keras.optmizers.Adam(
                                    learning_rate = 0.001, # Taxa de aprendizagem, quanto maior, mais rápido e menos preciso
                                    clipvalue = 0.5        # limita o valor máximo q um peso pode alcançar,
                                    )                      #  isso ajuda quando os pesos estão altos
                                                           #  porém os resultados ruins
rede_neural.compile(
    optimizer = otimizador,            
    loss = 'binary_crossentropy',  
    metrics = ['binary_accuracy'] 
    )
#--------------                    
              
rede_neural.fit(
    X_treinamento,   
    y_treinamento,   
    batch_size = 10, 
    epochs = 100)
                    
previsoes = rede_neural.predict(X_teste) 
previsoes = previsoes > 0.5              

accuracy_score(y_teste, previsoes)       
confusion_matrix(y_teste, previsoes)   
  
rede_neural.evaluate(X_teste, y_teste) 


#-------------- Visualizando os Pesos

pesos0 = rede_neural.layers[0].get_weights()
len(pesos0) # retorna 2 array, 1º Pesos da camada de entrada para a camada oculta, 2º Pesos do Bias
pesos1 = rede_neural.layers[1].get_weights()
pesos2 = rede_neural.layers[2].get_weights()
