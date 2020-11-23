from abc import ABCMeta,abstractmethod
import pandas as pd
import numpy as np
from Datos import *
from EstrategiaParticionado import *
from Clasificador import *
from Distancia import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable
import matplotlib.pyplot as plt



diabetes = Datos('./ConjuntoDatos/pima-indians-diabetes.csv')
wdbc = Datos('./ConjuntoDatos/pima-indians-diabetes.csv')
# Validacion simple con 100 iteraciones
vs = ValidacionSimple(0.30, 10)

datasets = [('diabetes', diabetes), ('wdbc', wdbc)]

distancias = []
de = DistanciaEuclidea()
dt = DistanciaManhattan()
dm = DistanciaMahalanobis()
distancias.append(de)
distancias.append(dt)
distancias.append(dm)
# Vecinos a utilizar
vecinos = [1, 3, 5, 11, 21]
# Normalizacion
normalizar = [True, False]

for d in datasets:
    t = PrettyTable()
    t.add_row([" ", " ", " ", " ", " ", " ", d[0]])
    t.add_row(["Vecinos", str(distancias[0]), " ", str(distancias[1]), " ", str(distancias[2]), " "])
    t.add_row([" ", "Normalizado", " ", "Normalizado", " ", "Normalizado", " "])
    t.add_row([" ", "Si", "No", "Si", "No", "Si", "No"])
    for idk, k in enumerate(vecinos):
        error = [[[],[],[]],[[],[],[]]]
        desv = [[[],[],[]],[[],[],[]]]
        for n in range(2):
            for idnor, norm in enumerate(normalizar):
                for idd, d in enumerate(distancias):
                    clas = ClasificadorVecinosProximos(k=k, distancia=d, normalizar=norm)
                    error[idnor][idd].append(clas.validacion(particionado=vs, dataset=d[1]))
        
        for idnor, norm in enumerate(normalizar):
            for idd, d in enumerate(distancias):
                error_t = error[idnor][idd]
                error[idnor][idd] = np.mean(error_t)
                # desv[idnorm][idd] = np.std(error_t)
        t.add_row([str(k), str(error[0][0]), str(error[1][0]), str(error[0][1]), str(error[1][1]), str(error[0][2]), str(error[1][2])])
    print(t)