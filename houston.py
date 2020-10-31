#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:34:30 2020

@author: gustavo
"""

# Importar el dataset
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.2, seed=10)

# Set de entrenamiento
x_train.shape

# Set de prueba
x_test.shape

# Ver 1 registro
x_train[:1]

"""
ARQUITECTURA DE LA RED NEURONAL
"""


# Modelo de red neuronal
from keras.models import Sequential # Construir capa por capa el modelo de red
from keras.layers import Dense, Activation # Capa densa: en la cual las neuronas estan conectadas a neuronas de la siguiente capa


# Creacion del modelo que contiene 4 capas con diferentes neuronas cada una
modelo = Sequential()

modelo.add(Dense(13,input_dim =13, kernel_initializer='normal', activation='relu')) #Creacion de una capa con 13 neuronas

modelo.add(Dense(6, kernel_initializer='normal', activation='relu')) #Creacion de una capa con 6 neuronas

modelo.add(Dense(4, kernel_initializer='normal', activation='relu')) #Creacion de una capa con 4 neuronas

modelo.add(Dense(1, kernel_initializer='normal')) #Creacion de una capa con 1 neurona de salida

#Compilar el modelo
modelo.compile(loss='mean_squared_error',optimizer='adam', metrics=['mean_absolute_percentage_error'])

# Imprime el resumen del modelo
print(modelo.summary())

# Usar una funcion de keras para hacernos una idea grafica del modelo
from keras.utils import plot_model

plot_model(modelo, to_file='modelo.png', show_shapes=True)


"""
ENTRENAMIENTO
"""

x_val = x_train[300:,] # Tomaremos 300 valores
y_val = y_train[300:,]

#Entrenarlo
#Tamaño de datos: batch_size, epocas = iteraciones para que el algoritmo aprender
# Prueba 1: con error de 99%
modelo.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))

# Prueba 2:
# Como le dimos mas iteraciones, obtuvo mas tiempo de ir entendiendo la informacion
modelo.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val))

# Crear la validacion con el set de prueba

resultado = modelo.evaluate(x_test, y_test)



"""
VISUALIZAR EL DESEMPEÑO GRAFRICAMENTE
"""





















