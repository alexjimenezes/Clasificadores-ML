from abc import ABCMeta,abstractmethod
import pandas as pd
import numpy as np
from Datos import *
from EstrategiaParticionado import *
from Clasificador import *


if __name__ == "__main__":

    datos = Datos('german.data')
    vs = ValidacionSimple(0.30, 10)
    vc = ValidacionCruzada(80)
    nb = ClasificadorNaiveBayes()
    error_simple = nb.validacion(vs, datos)
    print("Error medio en particion con validación simple: " + str(error_simple))
    error_cruzada = nb.validacion(vc, datos)
    print("Error medio en particion con validación cruzada: " + str(error_cruzada))

    """     particiones_vs  = vs.creaParticiones(datos.datos)
    naive_bayes = ClasificadorNaiveBayes()
    naive_bayes.entrenamiento(datos.extraeDatos(particiones_vs[1].indicesTrain), datos.nominalAtributos, datos.diccionario)
    clasificacion = naive_bayes.clasifica(datos.extraeDatos(particiones_vs[1].indicesTest), datos.nominalAtributos, datos.diccionario)
    valores = datos.extraeDatos(particiones_vs[1].indicesTest)
    print(naive_bayes.error(valores, clasificacion)) """
    