#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import pygad
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

class AGsimple(BaseEstimator, ClassifierMixin):
    def __init__(self, N=150, generaciones=25, prob_mut= 1/30, prob_cruc=0.85):
        self.N=N
        self.generaciones = generaciones
        self.prob_mut = prob_mut
        self.prob_cruc=prob_cruc

    def forward(self, pesos, x):
        z = x @ pesos
        return 1.0 / (1.0 + np.exp(-np.clip(4.9 * z, -60, 60)))

    def fit(self, x, y):
        semilla = 19
        random.seed(semilla)
        np.random.seed(semilla)
        x, y = check_X_y(x, y)
        self.classes_ = np.unique(y)
        num_genes = x.shape[1]
        mutation_percent_genes = int(round(self.prob_mut * 100))

        def fitness_func(ga_instance, solution, solution_idx):
            calc = self.forward(solution, x)
            salida = (calc >= 0.5).astype(int)
            return float(np.mean(salida == y))

        self.ga_ = pygad.GA(num_generations=self.generaciones, num_parents_mating=max(2, int(self.N*0.20)), fitness_func=fitness_func, sol_per_pop=self.N,
            keep_elitism=5, num_genes=num_genes, init_range_low=-2, init_range_high=5, parent_selection_type="sss", crossover_type="two_points", 
            mutation_type="random", mutation_percent_genes=mutation_percent_genes, crossover_probability=self.prob_cruc, random_seed=19)

        self.ga_.run()
        self.best_solution_, self.best_fitness_, _ = self.ga_.best_solution()
        return self

    def predict(self, x):
        check_is_fitted(self, attributes=["best_solution_"])
        x = check_array(x)
        calc = self.forward(self.best_solution_, x)
        return (calc>=0.5).astype(int)

