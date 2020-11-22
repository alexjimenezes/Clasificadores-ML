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
import matplotlib.pyplot as plt


if __name__ == "__main__":

    dataset = Datos('./ConjuntoDatos/wdbc.csv')
    # Validacion simple con 10 iteraciones
    vs = ValidacionSimple(0.30, 10)

    rlog = ClasificadorRegresionLogistica()
    de = DistanciaEuclidea()
    knn = ClasificadorVecinosProximos(k=11, distancia=de)
    nb = ClasificadorNaiveBayes()

    ROCpoint = nb.espacioROC(particionado=vs, dataset=dataset)
    plt.plot(ROCpoint[1],ROCpoint[0], 'o')
    plt.title("Espacio ROC")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
