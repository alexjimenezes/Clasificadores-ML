from abc import ABCMeta, abstractmethod
import numpy as np
from multipledispatch import dispatch
from scipy.stats import norm

from Distancia import *
import math
from random import choices, randint, random

MEDIA_V = 0
DESVIACION_V = 1


class Clasificador(object, metaclass=ABCMeta):

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass

    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass

        # Obtiene el numero de aciertos y errores para calcular la tasa de fallo

    def error(self, datos, pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        contador = 0
        # print("***********************")
        for a, b in zip(datos[:, -1], pred):
            # print(str(a) + "----" + str(b))
            if a == b:
                contador += 1
        error_prediccion = 1 - contador / len(pred)
        # print(error_prediccion)
        return error_prediccion

        # Realiza una clasificacion utilizando una estrategia de particionado determinada

    def validacion(self, particionado, dataset, seed=None):

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
        particiones_idx = particionado.creaParticiones(dataset.datos)

        error = []
        for i in range(len(particiones_idx)):
            particion_train = dataset.extraeDatos(particiones_idx[i].indicesTrain)
            particion_test = dataset.extraeDatos(particiones_idx[i].indicesTest)
            self.entrenamiento(particion_train, dataset.nominalAtributos, dataset.diccionario)
            prediccion = self.clasifica(particion_test, dataset.nominalAtributos, dataset.diccionario)
            error.append(self.error(particion_test, prediccion))

        media = np.average(error)
        return media

    @abstractmethod
    def reset_clasificador(self):
        pass

    # Obtiene la representación en el espacio ROC del modelo que ha afirmado un vector de predicciones "pred"

    def espacioROC(self, particionado, dataset, seed=None):
        particiones_idx = particionado.creaParticiones(dataset.datos)
        TPR = 0
        FPR = 0
        numPart = len(particiones_idx)
        for i in range(numPart):
            particion_train = dataset.extraeDatos(particiones_idx[i].indicesTrain)
            particion_test = dataset.extraeDatos(particiones_idx[i].indicesTest)
            self.entrenamiento(particion_train, dataset.nominalAtributos, dataset.diccionario)
            y = particion_test[:, -1]
            numClases = len(set(y))
            pred = self.clasifica(particion_test, dataset.nominalAtributos, dataset.diccionario)
            for j in range(numClases):
                TP = TN = FP = FN = 0
                for p, r in zip(pred, y):
                    TP += np.sum(p == r and r == j) / len(particiones_idx)
                    TN += np.sum(r != j and p != j) / len(particiones_idx)
                    FP += np.sum(p != r and p == j) / len(particiones_idx)
                    FN += np.sum(p != j and r == j) / len(particiones_idx)
                TPR = TPR + TP / (TP + FN)
                FPR = FPR + FP / (FP + TN)
        return [TPR / (numClases * numPart), FPR / (numClases * numPart)]


##############################################################################

class ClasificadorNaiveBayes(Clasificador):

    def __init__(self, laplace=False):
        self.tablaNominales = dict()
        self.tablaContinuos = dict()
        self.apriori = dict()
        self.laplace = laplace

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        self.reset_clasificador()
        # Nombre de los atributos
        atributos = diccionario["Columnas"][:-1]
        # Nombre de la columna de clase
        clase = diccionario["Columnas"][-1]
        n_filas = datostrain.shape[0]
        n_atributos = len(atributos)
        n_clases = len(diccionario[clase])
        # Columna con todas las clases
        col_clases = datostrain[:, -1]

        for c in range(n_clases):
            self.apriori[c] = np.count_nonzero(col_clases == c) / n_filas

        # Separación por clases
        # Creamos diccionarios
        separado = []
        for i in range(n_clases):
            separado.append([])
        for i in range(n_filas):
            vector = datostrain[i, :]
            valor_clase = int(vector[-1])
            separado[valor_clase].append(vector)
        # Convert to a numpy array
        for i in range(n_clases):
            separado[i] = np.array(separado[i])

        # Creación de tablas de atributos nominales
        for i in range(n_atributos):

            # If it is a nominal atribute:
            # for table in clases_separadas:
            #   for row in table
            #     sumar 1 tabla_atributo[row[id_att], clase]
            if atributosDiscretos[i]:
                tipos = len(diccionario[atributos[i]])
                self.tablaNominales[i] = np.zeros((tipos, n_clases))
                for c in range(n_clases):
                    for row in separado[c]:
                        self.tablaNominales[i][int(row[i]), c] += 1
                # self.tablaNominalesLaPlace[i] = np.array(self.tablaNominales[i], copy=True)
                # Apply laplace where neccessary
                if self.laplace == True and np.any(self.tablaNominales[i] == 0):
                    self.tablaNominales[i] += 1

            # If it is continues
            # Add a table of |[media, mean]| x nclases
            # Fill up the tables
            else:
                self.tablaContinuos[i] = np.empty((2, n_clases))
                for c in range(n_clases):
                    self.tablaContinuos[i][MEDIA_V, c] = np.mean(separado[c][:, i])
                    self.tablaContinuos[i][DESVIACION_V, c] = np.std(separado[c][:, i])

    def clasifica(self, datostest, atributosDiscretos, diccionario):
        # Nombre de los atributos
        atributos = diccionario["Columnas"][:-1]
        # Nombre de la columna de clase
        clase = diccionario["Columnas"][-1]
        n_filas = datostest.shape[0]
        n_atributos = len(atributos)
        n_clases = len(diccionario[clase])

        predicciones = np.empty(n_filas, dtype=int)

        for row in range(n_filas):
            multiplicador = dict()
            for c in range(n_clases):
                # Iniciamos el producto con la prob a priori
                multiplicador[c] = self.apriori[c]
                # Acontinuación vamos calculando la verosimilitud por apriori
                for i in range(n_atributos):
                    if atributosDiscretos[i]:
                        value = int(datostest[row][i])
                        sum_class = np.sum(self.tablaNominales[i][:, c])
                        ocurrencia_value = self.tablaNominales[i][value, c]
                        try:
                            prob = ocurrencia_value / sum_class
                        except ZeroDivisionError:
                            prob = 0
                        multiplicador[c] *= prob
                    else:
                        value = datostest[row][i]
                        mean = self.tablaContinuos[i][MEDIA_V, c]
                        std = self.tablaContinuos[i][DESVIACION_V, c]
                        prob = norm.pdf(value, loc=mean, scale=std)
                        multiplicador[c] *= prob
            predicciones[row] = max(multiplicador, key=multiplicador.get)

        return predicciones

    def reset_clasificador(self):
        self.tablaNominales = dict()
        self.tablaContinuos = dict()
        self.apriori = dict()


class ClasificadorVecinosProximos(Clasificador):

    def __init__(self, k, distancia, normalizar=True):
        self.datosTrain = None
        self.k = k
        self.distancia = distancia
        self.normalizar = normalizar

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        self.reset_clasificador()
        # Coger todas las columnas de atributos
        datos = datostrain
        n_atts = datos.shape[1] - 1
        # Inicializar matriz de distancias

        # Normalizamos los datostrain usando la Z-Score
        for i in range(n_atts):
            media = np.mean(datos[:, i])
            varianza = np.std(datos[:, i])
            datos[:, i] = (datos[:, i] - media) / varianza

        self.datosTrain = datos

    def clasifica(self, datostest, atributosDiscretos, diccionario):

        n_rows_train = self.datosTrain.shape[0]
        n_rows_test = datostest.shape[0]
        n_atts = datostest.shape[1] - 1

        matrizDistancias = np.empty((n_rows_train, n_rows_test))

        # Normalizamos los datostest usando la Z-Score
        if self.normalizar:
            for i in range(n_atts):
                media = np.mean(datostest[:, i])
                varianza = np.std(datostest[:, i])
                datostest[:, i] = (datostest[:, i] - media) / varianza

        # Calculamos una matriz de distancias de todos los puntos del dataset de test con train
        V = np.cov(np.transpose(self.datosTrain[:, :-1]))
        VI = np.linalg.inv(V)
        for i in range(n_rows_train):
            for j in range(n_rows_test):
                p1 = self.datosTrain[i, :-1]
                p2 = datostest[j, :-1]
                if isinstance(self.distancia, DistanciaMahalanobis):
                    matrizDistancias[i, j] = self.distancia.calcular(p1, p2, VI=VI)
                else:
                    matrizDistancias[i, j] = self.distancia.calcular(p1, p2)

        prediccion = []
        # Calculamos los k vecinos mas cercanos a cada punto de test
        for i in range(matrizDistancias.shape[1]):
            col = matrizDistancias[:, i]
            closestNeighbors = np.argsort(col)[:self.k]
            # Cogemos las clases pertenecientes a dichos vecinos
            clases = []
            for elem in closestNeighbors:
                clases.append(self.datosTrain[elem, -1])
            # Devolvemos la clase más repetida
            prediccion.append(np.argmax(np.bincount(clases)))

        return np.array(prediccion)

    def reset_clasificador(self):
        self.datosTrain = None


class ClasificadorRegresionLogistica(Clasificador):

    # Vector de coeficientes que modificará el método de entrenamiento
    def __init__(self, step=0.0001, epocas=200):
        self.coef = None
        self.step = step
        self.epocas = epocas

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        self.reset_clasificador()
        n_rows_train = datostrain.shape[0]
        n_atts = datostrain.shape[1] - 1

        # inicializa los coeficientes del (-0'5, 0'5)
        self.coef = np.random.rand(n_atts) - 0.5

        # Normalizamos los datostest usando la Z-Score
        for i in range(n_atts):
            media = np.mean(datostrain[:, i])
            varianza = np.std(datostrain[:, i])
            datostrain[:, i] = (datostrain[:, i] - media) / varianza

        # Hacemos el descenso de gradiente
        for e in range(self.epocas):
            for r in range(n_rows_train):
                s = sigmoid(self.coef, datostrain[r, :-1])
                self.coef = self.coef - self.step * (s - datostrain[r, -1]) * datostrain[r, :-1]

    # Clasifica en funcion d ela salida de la funcion sigmoidal
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        n_rows_test = datosTest.shape[0]
        pred = np.empty(n_rows_test)

        for r in range(n_rows_test):
            if sigmoid(self.coef, datosTest[r, :-1]) > 0.5:
                pred[r] = 0
            else:
                pred[r] = 1

        return pred

    def reset_clasificador(self):
        self.coef = None

    # Calcula la distancia mahalanobis entre dos puntos cualesquiera


# Calcula la funión sigmoidal sobre los coef y el vector indicado
# Devuelve un valor entre 0 y 1
def sigmoid(w, x):
    a = np.dot(w, x)
    e = np.exp(np.negative(a))
    return 1 / (1 + e)


# Clasificador utilizando algoitmo genetico
#   FFitness = (% acierto)^2
#   Elitismo = 5 %
#   Seleccion progenitores por método ruleta
#   Mutacion ----
#   Crossover con 2 descendientes

ELITE = 0.05
# Prob de mutacion = 0.5 %
MUTA = 0.05

class AlgoritmoGenetico(Clasificador):
    
    def __init__(self, tamano_poblacion, epocas):
        self.epocas = epocas
        self.tamano_poblacion = tamano_poblacion
        # N total de atts
        self.n_atts = None
        # Lista de longitud de cada lista de att_features_val
        self.att_features = None
        self.datosTrain = None
        # Lista con lista de valores posibles de cada att ordenados
        self.att_features_vals = None
        self.poblacion = None
        self.best_rule = None

    """
    Genera un individuo de longitud l
    """
    def genera_individuo(self, longitud):
        return choices([0, 1], k=longitud)

    """
    Genera una ppblacion de individuos de longitud l
    y tamaño especificado en el clasificador
    """
    def genera_poblacion(self, longitud):
        return [self.genera_individuo(longitud) for _ in range(self.tamano_poblacion)]

    """
    Calcular Fitness
        1. Clasifica con todas las reglas
        2. Calcula el porcentaje de acierto
        3. Lo eleva al cuadrado
    """
    def calcular_fitness(self, datos_train):
        clasificacion = self.clasifica_todos(datos_train)
        y_train = datos_train[:, -1]
        accuracy = []
        for vector in clasificacion:
            suma = 0
            for a, b in zip(vector, y_train):
                if a == b:
                    suma += 1
            accuracy.append(suma / len(y_train))
        max_acc = np.max(accuracy)
        mean_acc = np.mean(accuracy)
        value = np.power(np.array(accuracy), 2)
        return value, mean_acc, max_acc

    """
    Regla: lista de [1,0]
    Individuo: lista de [1,0]
    Resultado:
        True si tras aplicar la regla hay n_attrs 1
        False en caso contrario
    """
    def aplicar_regla(self, regla, individuo):
        return sum([r & b for r, b in zip(regla, individuo)]) == self.n_atts

    """
    Verctor: Vector del dataset
    Return: Array de [1, 0] representando el vector
    """
    def convertir_a_individuo(self, vector):
        indi = []
        for att in range(len(vector)):
            # We realize that if a number
            # posicion = self.att_features_vals[att].index(vector[att])
            posicion = vector[att]
            # Posibilidad de usar diccionario creado al inicio para reducir coste de acceso y operaciones
            for i in range(self.att_features[att]):
                if i != posicion:
                    indi.append(0)
                else:
                    indi.append(1)
        return indi

    """
    Aplicamos una funcion de crossover en un punto"""
    def obtener_descendientes(self, p1, p2, longitud):
        # Seleccionamos pto de corte
        cross = randint(1, longitud - 2)
        desc1 = p1[:cross] + p2[cross:]
        desc2 = p2[:cross] + p1[cross:]
        return [desc1, desc2]

    def mutar_descendientes(self, descendienes, longitud):
        for des in descendienes:
            if random() <= MUTA:
                pos = randint(0, longitud - 1)
                # Cambia 0 por 1 y 1 por 0
                des[pos] = abs(des[pos] - 1)
        return descendienes

    """"
    Entrena el alforitmo creando sucesivas poblaciiones,
    Aplicando cruces de progenitores para crear hijos,
    Mutar hijos y elegir elite que pasa automáticamente.
    """
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        print("Comenzando entrenamiento:\n\n")
        self.reset_clasificador()
        self.n_atts = datosTrain.shape[1] - 1
        # Will hold the number of features per attribute
        self.att_features = []
        self.att_features_vals = []
        # Number of individuals that pass automatically to next generation
        select_elite = math.floor(ELITE * self.tamano_poblacion)
        # Total number of features
        total_features = 0
        for i in range(self.n_atts):
            new = sorted(list(set(datosTrain[:, i])))
            self.att_features_vals.append(new)
            features = len(new)
            total_features += features
            self.att_features.append(features)

        # Generamos una poblacion
        self.poblacion = self.genera_poblacion(total_features)
        for i in range(self.epocas):
            # Obtenemos los pesos de la poblacion guardada en self.poblacion
            val, mean, maxi = self.calcular_fitness(datosTrain)
            if (i + 1) % 5 == 0:
                print("Epoca " + str(i) + ": \tAcierto medio = {0:.4f}\t Acierto Max = {1:.4f}\n".format(mean, maxi))
            # Convertimos los pesos en una lista con suma = 1
            pesos = val / np.sum(val)
            nueva_poblacion = []
            # Creamos poblacion descendiente
            while len(nueva_poblacion) < self.tamano_poblacion - select_elite - 1:
                # Elegimos progenitores utilizando el metodo de la ruleta
                # Es importante que sea sin reemplazamiento para que los progentiores sean individuos distintos
                prog_idx = np.random.choice(list(range(self.tamano_poblacion)), size=2, replace=False, p=pesos)
                prog1 = self.poblacion[prog_idx[0]]
                prog2 = self.poblacion[prog_idx[1]]
                nueva_poblacion += self.obtener_descendientes(prog1, prog2, total_features)
            # Mutamos los descendientes directos
            nueva_poblacion = self.mutar_descendientes(nueva_poblacion, total_features)
            # Añadimos la elite
            elite = [indi for _, indi in sorted(zip(val, self.poblacion), reverse=True)][:self.tamano_poblacion - len(nueva_poblacion)]
            nueva_poblacion += elite
            # Escpgemos la mejor regla
            self.best_rule = elite[0]
            # sustituimos poblacion y volvemos a empezar
            self.poblacion = nueva_poblacion

        print("El mejor individuo es: " + str(self.best_rule) + "\n")

    """
    Clasifica
    Clasifica utilizando todo el conjunto de reglas
    Devuelve una lista de listas que contiene la clasificación de cada regla
    """
    def clasifica_todos(self, datosTest):
        resultado = []
        for regla in self.poblacion:
            resultado_regla = []
            for vector in datosTest:
                indi = self.convertir_a_individuo(vector[:-1])
                resultado_regla.append(self.aplicar_regla(regla, indi))
            resultado.append(resultado_regla)
        return resultado

    """
    Clasifica
    Overload del método anterior que permite pasar una regla al método que se utilizará para clasificar.
    Devuelve la clasificación del dataset con esa regla
    """
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        resultado = []
        for vector in datosTest:
            indi = self.convertir_a_individuo(vector[:-1])
            resultado.append(int(self.aplicar_regla(self.best_rule, indi)))
        return resultado

    # Funcion que reinicia la instancia de clase
    def reset_clasificador(self):
        self.__init__(self.tamano_poblacion, self.epocas)


