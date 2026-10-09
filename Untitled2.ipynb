{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "600fc5e6-2661-4f23-9b99-2ccc676e730b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "import numpy as np\n",
    "import pygad\n",
    "from sklearn.base import BaseEstimator, ClassifierMixin\n",
    "from sklearn.utils.validation import check_is_fitted, check_X_y, check_array\n",
    "\n",
    "class AGsimple(BaseEstimator, ClassifierMixin):\n",
    "    def __init__(self, N=150, generaciones=25, prob_mut= 1/30, prob_cruc=0.85):\n",
    "        self.N=N\n",
    "        self.generaciones = generaciones\n",
    "        self.prob_mut = prob_mut\n",
    "        self.prob_cruc=prob_cruc\n",
    "\n",
    "    def forward(self, pesos, x):\n",
    "        z = x @ pesos\n",
    "        return 1.0 / (1.0 + np.exp(-np.clip(4.9 * z, -60, 60)))\n",
    "\n",
    "    def fit(self, x, y):\n",
    "        semilla = 19\n",
    "        random.seed(semilla)\n",
    "        np.random.seed(semilla)\n",
    "        x, y = check_X_y(x, y)\n",
    "        self.classes_ = np.unique(y)\n",
    "        num_genes = x.shape[1]\n",
    "        mutation_percent_genes = int(round(self.prob_mut * 100))\n",
    "\n",
    "        def fitness_func(ga_instance, solution, solution_idx):\n",
    "            calc = self.forward(solution, x)\n",
    "            salida = (calc >= 0.5).astype(int)\n",
    "            return float(np.mean(salida == y))\n",
    "            \n",
    "        self.ga_ = pygad.GA(num_generations=self.generaciones, num_parents_mating=max(2, int(self.N*0.20)), fitness_func=fitness_func, sol_per_pop=self.N,\n",
    "            num_genes=num_genes, init_range_low=-2, init_range_high=5, parent_selection_type=\"sss\", keep_parents=0, crossover_type=\"two_points\", \n",
    "            mutation_type=\"random\", mutation_percent_genes=mutation_percent_genes, crossover_probability=self.prob_cruc, random_seed=19)\n",
    "        self.ga_.run()\n",
    "        self.best_solution_, self.best_fitness_, _ = self.ga_.best_solution()\n",
    "        return self\n",
    "        \n",
    "    def predict(self, x):\n",
    "        check_is_fitted(self, attributes=[\"best_solution_\"])\n",
    "        x = check_array(x)\n",
    "        calc = self.forward(self.best_solution_, x)\n",
    "        return (calc>=0.5).astype(int)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
