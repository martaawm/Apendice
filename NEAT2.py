#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import random

class Nodo:
    def __init__(self, num_nodo, tipo_nodo):
        self.num = num_nodo    
        self.tipo = tipo_nodo  
        self.capa = 0          
        self.valor_activacion = 0.0 

class Conexion:
    def __init__(self, nodo_entrada, nodo_salida, peso, bit_act, num_innovacion):
        self.nodo_entrada = nodo_entrada    
        self.nodo_salida = nodo_salida     
        self.peso = peso
        self.bit_act = bit_act       
        self.num_innovacion = num_innovacion  

class Genotipo:
    def __init__(self):
        self.nodos = {}       
        self.conexiones = {} 
        self.fitness = 0.0
        self.fitness_ajustado = 0.0

    def calcular_capas(self):
        for n in self.nodos.values():
            if n.tipo != 'entrada':
                n.capa = -1
        a = True
        max_iter = len(self.nodos) 
        iteraciones = 0
        while a and iteraciones < max_iter:
            a = False
            iteraciones += 1
            for c in self.conexiones.values():
                if c.bit_act:
                    capa_entrada = self.nodos[c.nodo_entrada].capa
                    capa_salida = self.nodos[c.nodo_salida].capa
                    if capa_entrada != -1:
                        nueva = capa_entrada + 1
                        if nueva > self.nodos[c.nodo_salida].capa:
                            self.nodos[c.nodo_salida].capa = nueva
                            a = True

    def fenotipo(self, entradas):
        ids_entrada = [n.num for n in self.nodos.values() if n.tipo == 'entrada']
        ids_entrada.sort()
        if len(entradas) != len(ids_entrada):
            raise ValueError(f" Debe haber {len(ids_entrada)} entradas")
        for idx, val in enumerate(entradas):
            self.nodos[ids_entrada[idx]].valor_activacion = val
        self.calcular_capas()

        nodos_ordenados = sorted(self.nodos.values(), key=lambda n: n.capa)
        for n in nodos_ordenados:
            if n.tipo == 'entrada':
                continue            
            z = 0.0
            for c in self.conexiones.values():
                if c.bit_act and c.nodo_salida == n.num:
                    z += self.nodos[c.nodo_entrada].valor_activacion * c.peso
            n.valor_activacion = 1.0 / (1.0 + math.exp(-4.9 * z))

        ids_salida = [n.num for n in self.nodos.values() if n.tipo == 'salida']
        ids_salida.sort()
        return [self.nodos[i].valor_activacion for i in ids_salida]

    @staticmethod
    def distancia_genetica(g1, g2, c1=1.0, c2=1.0, c3=0.4):
        innovaciones_g1 = set(g1.conexiones.keys())
        innovaciones_g2 = set(g2.conexiones.keys())

        if not innovaciones_g1 and not innovaciones_g2:
            return 0.0

        max1 = max(innovaciones_g1) if innovaciones_g1 else 0    
        max2 = max(innovaciones_g2) if innovaciones_g2 else 0

        coincidentes = innovaciones_g1.intersection(innovaciones_g2)
        disjuntos = 0
        excesivos = 0
        diferencia_pesos = 0.0

        for i in coincidentes:
            diferencia_pesos += abs(g1.conexiones[i].peso - g2.conexiones[i].peso)

        innovaciones_dif = innovaciones_g1.symmetric_difference(innovaciones_g2)
        for i in innovaciones_dif:
            if i > max2 if i in innovaciones_g1 else i > max1: 
                excesivos += 1
            else:
                disjuntos += 1

        N = max(len(g1.conexiones), len(g2.conexiones)) 
        if N < 20:
            N = 1.0

        W = (diferencia_pesos / len(coincidentes)) if coincidentes else 0.0
        return (c1 * excesivos / N) + (c2 * disjuntos / N) + (c3 * W)

    @staticmethod
    def cruzar(p1, p2, prob_desactivado=0.75):
        if p1.fitness > p2.fitness:
            p_mejor, p_peor = p1, p2
        elif p2.fitness > p1.fitness:
            p_mejor, p_peor = p2, p1
        else:
            p_mejor, p_peor = (p1, p2) if random.random() < 0.5 else (p2, p1)

        hijo = type(p_mejor)()

        for n in p_mejor.nodos.values():
            hijo.nodos[n.num] = Nodo(n.num, n.tipo)

        for ni, con_mejor in p_mejor.conexiones.items():
            if ni in p_peor.conexiones:
                con_coin = con_mejor if random.random() < 0.5 else p_peor.conexiones[ni] 
                bit_act = con_coin.bit_act
                if not con_mejor.bit_act or not p_peor.conexiones[ni].bit_act:
                    if random.random() < prob_desactivado:
                        bit_act = False
                    else:
                        bit_act = True
                hijo.conexiones[ni] = Conexion(con_coin.nodo_entrada, con_coin.nodo_salida, con_coin.peso, bit_act, ni)
            else:
                hijo.conexiones[ni] = Conexion(con_mejor.nodo_entrada, con_mejor.nodo_salida, con_mejor.peso, con_mejor.bit_act, ni)
        return hijo

    def mutar_peso(self, p_mutacion=0.8, p_uniforme=0.9):
        if random.random() < p_mutacion:
            for c in self.conexiones.values():
                if random.random() < p_uniforme:
                    c.peso += random.uniform(-0.5, 0.5)
                    if c.peso > 8.0:
                        c.peso = 8.0
                    elif c.peso < -8.0:
                        c.peso = -8.0
                else:
                    c.peso = random.uniform(-1.0, 1.0)

    def mutar_conexion(self, poblacion_cont, Iter=20):
        self.calcular_capas()
        nodos_ids = list(self.nodos.keys())
        for _ in range(Iter):
            n1 = random.choice(nodos_ids)
            n2 = random.choice(nodos_ids)
            if n1 == n2: 
                continue 
            if self.nodos[n1].tipo == 'salida' or self.nodos[n2].tipo == 'entrada':
                continue 
            if self.nodos[n1].capa >= self.nodos[n2].capa:
                continue 

            e = any(c.nodo_entrada == n1 and c.nodo_salida == n2 for c in self.conexiones.values())
            if not e:
                inn = poblacion_cont.calculo_innovacion(n1, n2)
                peso = random.uniform(-1.0, 1.0)
                self.conexiones[inn] = Conexion(n1, n2, peso, True, inn)
                break

    def mutar_nodo(self, poblacion_cont):
        self.calcular_capas()
        conexiones_activas = [c for c in self.conexiones.values() if c.bit_act]
        if not conexiones_activas:
            return
        c = random.choice(conexiones_activas)
        c.bit_act = False
        nuevo_id = max(self.nodos.keys()) + 1
        self.nodos[nuevo_id] = Nodo(nuevo_id, 'oculta')

        inn1 = poblacion_cont.calculo_innovacion(c.nodo_entrada, nuevo_id)
        self.conexiones[inn1] = Conexion(c.nodo_entrada, nuevo_id, 1.0, True, inn1)

        inn2 = poblacion_cont.calculo_innovacion(nuevo_id, c.nodo_salida)
        self.conexiones[inn2] = Conexion(nuevo_id, c.nodo_salida, c.peso, True, inn2)

class Especie:
    def __init__(self, representante):
        self.representante = representante
        self.miembros = [representante]
        self.mejor_fitness_his = 0.0
        self.estancada = 0

    def actualizar(self):
        max_fit = max(m.fitness for m in self.miembros)
        if max_fit > self.mejor_fitness_his:
            self.mejor_fitness_his = max_fit
            self.estancada = 0
        else:
            self.estancada += 1
        self.representante = random.choice(self.miembros)
        self.miembros = []

class Poblacion:
    def __init__(self, N, genotipos, num_entrada, num_salida, 
                 prob_mutar_peso=0.8, prob_mutar_conexion=0.05, prob_mutar_nodo=0.03):
        self.N = N
        self.especies = []
        self.individuos = []
        self.contador = 0
        self.historico = {}

        self.prob_mutar_peso = prob_mutar_peso
        self.prob_mutar_conexion = prob_mutar_conexion
        self.prob_mutar_nodo = prob_mutar_nodo

        for _ in range(N):
            g = genotipos()
            for i in range(num_entrada):
                g.nodos[i] = Nodo(i, 'entrada')
            for k in range(num_salida):
                g.nodos[num_entrada + k] = Nodo(num_entrada + k, 'salida')
            for i in range(num_entrada):
                for k in range(num_salida):
                    inn = self.calculo_innovacion(i, num_entrada + k)
                    g.conexiones[inn] = Conexion(i, num_entrada + k, random.uniform(-1.0, 1.0), True, inn)
            self.individuos.append(g)

    def calculo_innovacion(self, n_entrada, n_salida):
        if (n_entrada, n_salida) in self.historico:
            return self.historico[(n_entrada, n_salida)]
        self.contador += 1
        self.historico[(n_entrada, n_salida)] = self.contador
        return self.contador

    def especiacion(self, delta_t=3.0):
        for i in self.individuos:
            asignado = False
            for esp in self.especies:
                if Genotipo.distancia_genetica(i, esp.representante) <= delta_t:
                    esp.miembros.append(i)
                    asignado = True
                    break
            if not asignado:
                nueva_esp = Especie(i)
                self.especies.append(nueva_esp)
        self.especies = [e for e in self.especies if e.miembros]

    def reproduccion(self):
        for esp in self.especies:
            num_miembros = len(esp.miembros)
            for i in esp.miembros:
                i.fitness_ajustado = i.fitness / num_miembros

        self.especies = [e for e in self.especies if e.estancada < 15 or e.representante == max(self.individuos, key=lambda x: x.fitness)]

        suma_ajustada_global = sum(i.fitness_ajustado for i in self.individuos)
        nueva_poblacion = []

        if suma_ajustada_global == 0:
            nueva_poblacion = []
            for _ in range(self.N):
                p = random.choice(self.individuos)
                nueva_poblacion.append(Genotipo.cruzar(p, p))
            self.individuos = nueva_poblacion
            return

        for esp in self.especies:
            suma_esp = sum(m.fitness_ajustado for m in esp.miembros)
            asig = int((suma_esp / suma_ajustada_global) * self.N)

            if asig == 0:
                continue
            esp.miembros.sort(key=lambda x: x.fitness, reverse=True)
            num_selec = max(1, int(len(esp.miembros) * 0.2))
            selec = esp.miembros[:num_selec]

            for _ in range(asig):
                if random.random() < 0.75:
                    p1 = random.choice(selec)
                    p2 = random.choice(selec) if random.random() > 0.001 else random.choice(self.individuos)
                    hijo = Genotipo.cruzar(p1, p2)
                else:
                    p = random.choice(selec)
                    hijo = Genotipo.cruzar(p, p)  
                hijo.mutar_peso(p_mutacion=self.prob_mutar_peso)
                if random.random() < self.prob_mutar_conexion:
                    hijo.mutar_conexion(self)
                if random.random() < self.prob_mutar_nodo:
                    hijo.mutar_nodo(self)
                nueva_poblacion.append(hijo)

            esp.actualizar()

        while len(nueva_poblacion) < self.N:
            p_alt = random.choice(self.individuos)
            nueva_poblacion.append(Genotipo.cruzar(p_alt, p_alt))
        self.individuos = nueva_poblacion

