#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

class Estructura(nn.Module):
    def __init__(self, entrada_dim):
        super().__init__()
        self.fc = nn.Linear(entrada_dim, 1, bias=False)

    def forward(self, x):
        z = self.fc(x)
        return torch.sigmoid(4.9 * z)

class Retropropagacion(BaseEstimator, ClassifierMixin):
    def __init__(self, epocas=25, lr=0.01, batches=1):
        self.epocas = epocas
        self.lr = lr
        self.batches = batches

    def fit(self, x, y):
        semilla = 19
        random.seed(semilla)
        np.random.seed(semilla)
        torch.manual_seed(semilla)
        x, y = check_X_y(x, y)
        self.classes_ = np.unique(y)
        self.model_ = Estructura(x.shape[1])
        optimizer = optim.SGD(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        n = x_t.shape[0]
        tam_batch = max(1, n // self.batches)
        self.model_.train()
        for e in range(self.epocas):
            i = torch.randperm(n)
            x_reor = x_t[i]
            y_reor = y_t[i]
            perdida_epoca = 0.0
            for j in range(0, n, tam_batch):
                x_b = x_reor[j:j + tam_batch]
                y_b = y_reor[j:j + tam_batch]
                optimizer.zero_grad()
                salida = self.model_(x_b)
                perdida = criterion(salida, y_b)
                perdida.backward()
                optimizer.step()
                perdida_epoca += perdida.item() * x_b.shape[0]
        return self

    def predict(self, x):
        check_is_fitted(self, attributes=["model_"])
        x = check_array(x)
        self.model_.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32)
            salida = self.model_(x_t).reshape(-1).numpy()
        return (salida>=0.5).astype(int)

