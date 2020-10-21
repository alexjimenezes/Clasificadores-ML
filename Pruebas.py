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
    naive_bayes.entrenamiento(datos.extraeDatos(particiones_vs[0].indicesTrain), datos.nominalAtributos, datos.diccionario)
