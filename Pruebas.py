from abc import ABCMeta,abstractmethod
import pandas as pd
import numpy as np
from Datos import *
from EstrategiaParticionado import *
from Clasificador import *


if __name__ == "__main__":

    datos = Datos('./ConjuntoDatos/tic-tac-toe.data')
    vs = ValidacionSimple(0.30, 5)
    particiones_vs  = vs.creaParticiones(datos.datos)
    naive_bayes = ClasificadorNaiveBayes()
    naive_bayes.entrenamiento(datos.extraeDatos(particiones_vs[1].indicesTrain), datos.nominalAtributos, datos.diccionario)
    clasificacion = naive_bayes.clasifica(datos.extraeDatos(particiones_vs[1].indicesTest), datos.nominalAtributos, datos.diccionario)
    valores = datos.extraeDatos(particiones_vs[1].indicesTest)[:, -1]
    contador = 0
    for a, b in zip(valores, clasificacion):
        if a == b:
            contador += 1
    print(contador/len(clasificacion))