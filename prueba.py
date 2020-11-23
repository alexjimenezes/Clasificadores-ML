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

    wdbc = Datos('./ConjuntoDatos/wdbc.csv')
    diabetes = Datos('./ConjuntoDatos/pima-indians-diabetes.csv')

    datasets = {'diabetes': diabetes, 'wdbc': wdbc}
    # Validacion simple con 10 iteraciones
    vs = ValidacionSimple(0.30, 10)

    rlog = ClasificadorRegresionLogistica()
    de = DistanciaEuclidea()
    knn = ClasificadorVecinosProximos(k=11, distancia=de)
    nb = ClasificadorNaiveBayes()

    clasificadores = {"Regresión logística":rlog, "K vecinos próximos":knn, "Naive bayes":nb}

    for (k_dataset, v_dataset) in datasets.items():
        for (k_clasificador, v_clasificador) in clasificadores.items():
            ROCpoint = [0, 0]
            for i in range(1):
                ROCpoint_temp = v_clasificador.espacioROC(particionado=vs, dataset=v_dataset)
                ROCpoint[0] = ROCpoint_temp[0]
                ROCpoint[1] = ROCpoint_temp[1]
            plt.plot(ROCpoint[1],ROCpoint[0], label=k_clasificador)
        plt.title("Espacio ROC del dataset "+k_dataset)
        print(k_dataset)
        print(k_clasificador)
        plt.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        plt.legend(loc="lower right", title="Nombre de los modelos", frameon=False)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()


