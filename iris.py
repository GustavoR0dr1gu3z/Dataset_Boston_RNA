#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:47:34 2020

@author: gustavo
"""

import numpy as np # LIbreria para operaciones matematicas
import matplotlib.pyplot as plt # Para graficos

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron 

iris = load_iris()

iris.target # Mostrar las muestras a que especie de iris corresponde
iris.data[:5,:] # Traer datos especificos

data = iris.data[:,(2,3)]
labels = iris.target

plt.figure(figsize=(13,6))
plt.scatter(data[:,0],data[:,1], c=labels)
plt.show


# Variable objetivo
y = (iris.target == 2).astype(np.int) # iris virginica, que sea entero

# Mandamos a llamar al metodo perceptron
test_perceptron = Perceptron()

# Vamos a entrenar al perceptron con fit = metodo con el cual ajustamos
test_perceptron.fit(data,y) # Valores de entrada, Variable objetivo

# Evaluar su desempeño, usando el test con el metodo predict, con la muestra 5.1,2
y1_pred = test_perceptron.predict([[5.1,2]])
print("La prediccion 1: ", y1_pred)















