from abc import ABCMeta,abstractmethod
import pandas as pd
import numpy as np
from Datos import *
from EstrategiaParticionado import *
from Clasificador import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # dataset = Datos('./ConjuntoDatos/pima-indians-diabetes.csv')
    # dataset = Datos('./ConjuntoDatos/german.data')
    dataset = Datos('./ConjuntoDatos/tic-tac-toe.data')
    # dataset = Datos('./ConjuntoDatos/pima-indians-diabetes.csv')
    # Validacion simple con 100 iteraciones
    vs = ValidacionSimple(0.30, 10)

    # Creación del clasificador con d euclidea
    knn = CalsificadorVecinosProximos()
    de = DistanciaEuclidea()

    # Coge una particion de muestra
    particiones_idx = vs.creaParticiones(dataset.datos)

    error = 0

    for particion in particiones_idx:
        particion_train = dataset.extraeDatos(particion.indicesTrain)
        particion_test = dataset.extraeDatos(particion.indicesTest)
        # Entrena el clasificador
        knn.entrenamiento(particion_train)
        # Clasifica
        pred = knn.clasifica(particion_test, de, k=11)
        # Calculamos el error
        error += knn.error(particion_test, pred)

    print(error/len(particiones_idx))