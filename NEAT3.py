#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from ipynb.fs.full.NEAT2 import Genotipo, Poblacion

class ClasificacionGenotipo(Genotipo):

    def __init__(self, num_entradas=None, num_salidas=1):
        super().__init__()
        self.num_entradas = num_entradas
        self.num_salidas = num_salidas
        self.fitness = 0.0

    def inicializar(self, num_entradas, num_salidas=1):
        self.num_entradas = num_entradas
        self.num_salidas = num_salidas
        return self

    def predecir(self, entrada):
        salida = self.fenotipo(entrada)
        return 1 if salida[0] >= 0.5 else 0

    def calcular_fitness(self, x, y):
        aciertos = 0
        for i in range(len(x)):
            prediccion = self.predecir(x[i])
            if prediccion == y[i]:
                aciertos += 1
        self.fitness = aciertos / len(x)
        return self.fitness

class NEAT(BaseEstimator, ClassifierMixin):
    def __init__(self, N=150, generaciones=25, delta_t=3.0, prob_mutar_peso=0.8, prob_mutar_conexion=0.05, prob_mutar_nodo=0.03):
        self.N = N
        self.generaciones = generaciones
        self.delta_t = delta_t
        self.prob_mutar_peso = prob_mutar_peso
        self.prob_mutar_conexion = prob_mutar_conexion
        self.prob_mutar_nodo = prob_mutar_nodo

    def fit(self, x, y):
        semilla = 19
        random.seed(semilla)
        np.random.seed(semilla)
        x, y = check_X_y(x, y)
        self.classes_ = np.unique(y)
        poblacion = Poblacion(N=self.N,genotipos=lambda: ClasificacionGenotipo(num_entradas=x.shape[1], num_salidas=1), num_entrada=x.shape[1], 
                              num_salida=1, prob_mutar_peso=self.prob_mutar_peso, prob_mutar_conexion=self.prob_mutar_conexion, prob_mutar_nodo=self.prob_mutar_nodo)
        self.mejor_fitnesshis_ = []
        for generacion in range(self.generaciones):
            for i in poblacion.individuos:
                i.calcular_fitness(x, y)
            poblacion.especiacion(delta_t=self.delta_t)
            mejor_fit = max(ind.fitness for ind in poblacion.individuos)
            self.mejor_fitnesshis_.append(mejor_fit)
            if generacion < self.generaciones - 1:
                poblacion.reproduccion()
        self.poblacion_ = poblacion
        self.mejor_ = max(poblacion.individuos, key=lambda j: j.fitness)
        self.fitness_ = self.mejor_.fitness
        return self

    def predict(self, x):
        check_is_fitted(self, attributes=["mejor_"])
        x = check_array(x)
        return np.array([self.mejor_.predecir(j) for j in x])

