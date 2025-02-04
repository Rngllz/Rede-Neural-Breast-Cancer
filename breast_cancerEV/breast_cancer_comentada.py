

#-------------- Importação de Bibliotecas

import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential


#-------------- Configurando as Entradas e Saídas

# Entradas [PREVISORES]
# 30 características ( 30 colunas)
# 569 tumores ( 569 linhas)
X = pd.read_csv('C:/Users/User/Documents/GitHub/Rede-Neural-Breast-Cancer/base_de_dados/entradas_breast.csv')

# Saidas [RESULTADOS]
# 1 caracterísitca ( 1 coluna 'é maligno?' 0 = 'Benigno' 1 = 'Maligno')
# 569 resultados (569 linhas)
y = pd.read_csv('C:/Users/User/Documents/GitHub/Rede-Neural-Breast-Cancer/base_de_dados/saidas_breast.csv')


#-------------- Configurando o Treinamento e Teste

# Criação da base de treinamento e da base de teste                               25% para testar 75% para treinar
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.25)

# Treinamento, vai usar 75% dos dados (426) para treinar a rede neural
X_treinamento.shape # Previsores 426
y_treinamento.shape # Resultados 426

# Teste, vai usar 25% dos dados (143) para testar a rede neural
X_teste.shape # Previsores 143
y_teste.shape # Resultados 143    
                    

#-------------- Modelando a Rede Neural

             # Pq uma rede neural é uma sequência de camadas, uma conectada na outra
rede_neural = Sequential([
        # 1ª camada (entrada), 30 neurônicos (30 colunas)
        tf.keras.layers.InputLayer(shape = (30,)),
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
        # 2ª camada (oculta), 16 neurônios, função de ativação relu e pesos iniciados randomicamente 
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'),
        # 3ª camada (saída), 1 neurônio, função de ativação sigmoid
        tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])

rede_neural.summary() # Existe o BIAS, 1 ligando todos os neurônios da camada oculta e outro ligando o neurônio de saída
                    

#-------------- Configurando a Rede Neural

rede_neural.compile(
    optimizer = 'adam',            # melhoria na descida do gradiente estocástico
    loss = 'binary_crossentropy',  # Usado para calculo do erro, específico para resultados binários
    metrics = ['binary_accuracy']  # Usada para fazer a avaliação da rede, % de acerto
    )
                    

#-------------- Treinamento da Rede Neural ( sendo feito o teste com ela mesmo [não recomendado])
                
rede_neural.fit(
    X_treinamento,   # previsores 
    y_treinamento,   # resultados
    batch_size = 10, # será criado grupos de 10 em 10 registros, quando termina o batch atualiza os pesos
    epochs = 100)    # cada época, a rede neural processa todos os registros
                    

#-------------- Previsões com a Rede Neural usando a base de TESTE

previsoes = rede_neural.predict(X_teste) # Retorna em notação ciêntifica (probabilidade)
previsoes = previsoes > 0.5              # Transforma em True e False, quanto maior o valor (0.5) maior a confiança da rede

accuracy_score(y_teste, previsoes)       # Comparando respostas reais x previsões (utilizando a base de teste [recomendado])

confusion_matrix(y_teste, previsoes)     #              C0  C1  
                                         # Classe 0  [[ 48, 6  ],  48 tumores benignos corretos, 6 foram classificados errado
                                         # Classe 1   [  9, 80 ]]  80 tumores malignos corretos, 9 foram classificados errado
                                         #                         128 tumores corretos, 15 errados
                                         # 0=BNG 1=MLG             128 / 143 (total) = 0.89 (% acerto) 
                                         #                         15  / 143 (total) = 0.10 (% erro)
                                         
rede_neural.evaluate(X_teste, y_teste) # testar a rede (já treinada) com os 143 dados previsores e resultados 
