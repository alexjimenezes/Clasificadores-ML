from abc import ABCMeta,abstractmethod
import numpy as np
from scipy.stats import norm
import math

MEDIA_V = 0
DESVIACION_V = 1

class Clasificador(object, metaclass=ABCMeta):
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass 


  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  def error(self,datos,pred):
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
  def validacion(self,particionado,clasificador,dataset,seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
    particiones_idx = particionado.creaParticiones(dataset.datos)

    error_total = 0
    for i in range(len(particiones_idx)):
      particion_train = dataset.extraeDatos(particiones_idx[i].indicesTrain)
      particion_test = dataset.extraeDatos(particiones_idx[i].indicesTest)
      if clasificador["nombre"] == "naiveBayes":
        self.entrenamiento(particion_train, dataset.nominalAtributos, dataset.diccionario, laplace=clasificador["laplace"])
      prediccion = self.clasifica(particion_test, dataset.nominalAtributos, dataset.diccionario)
      error_total += self.error(particion_test, prediccion)
    
    error_total /= len(particiones_idx)
    return error_total
  
  @abstractmethod
  def reset_clasificador(self):
    pass

  #Obtiene la representación en el espacio ROC del modelo que ha afirmado un vector de predicciones "pred"
  
  def espacioROC(self,particionado,clasificador,dataset,seed=None):
    particiones_idx = particionado.creaParticiones(dataset.datos)
    TPR = 0
    FPR = 0
    numPart = len(particiones_idx)
    for i in range(numPart):
      
      particion_train = dataset.extraeDatos(particiones_idx[i].indicesTrain)
      particion_test = dataset.extraeDatos(particiones_idx[i].indicesTest)
      if clasificador["nombre"] == "naiveBayes":
        self.entrenamiento(particion_train, dataset.nominalAtributos, dataset.diccionario, laplace=clasificador["laplace"])
      y = particion_test[:, -1]
      numClases = len(set(y))
      pred = self.clasifica(particion_test, dataset.nominalAtributos, dataset.diccionario)
      for j in range(numClases):
        TP = TN = FP = FN = 0
        for p, r in zip(pred, y):       
          TP += np.sum(p ==  r and r == j)/len(particiones_idx)
          TN += np.sum( r != j and p != j)/len(particiones_idx)
          FP += np.sum(p != r and p == j)/len(particiones_idx)
          FN += np.sum(p != j and r == j)/len(particiones_idx)
        TPR = TPR + TP/(TP + FN)
        FPR = FPR + FP/(FP + TN)

    return [TPR/(numClases*numPart), FPR/(numClases*numPart)]

##############################################################################

class ClasificadorNaiveBayes(Clasificador):

  def __init__(self):
    self.tablaNominales = dict()
    self.tablaContinuos = dict()
    self.apriori = dict()


  def entrenamiento(self,datostrain,atributosDiscretos,diccionario, laplace=False):
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

    ##### Separación por clases
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

    ##### Creación de tablas de atributos nominales
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
        #self.tablaNominalesLaPlace[i] = np.array(self.tablaNominales[i], copy=True)
        # Apply laplace where neccessary
        if laplace == True and np.any(self.tablaNominales[i] == 0):
          self.tablaNominales[i] += 1
      
      # If it is continues
      # Add a table of |[media, mean]| x nclases
      # Fill up the tables
      else:
        self.tablaContinuos[i] = np.empty((2, n_clases))
        for c in range(n_clases):
          self.tablaContinuos[i][MEDIA_V, c] = np.mean(separado[c][:, i])
          self.tablaContinuos[i][DESVIACION_V, c] = np.std(separado[c][:, i])


  def clasifica(self,datostest,atributosDiscretos,diccionario):
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

class CalsificadorVecinosProximos(Clasificador):

  def __init__(self):
    self.datosTrain = None

  def entrenamiento(self,datostrain,atributosDiscretos=None,diccionario=None):

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

  def clasifica(self,datostest,distancia,k=5,atributosDiscretos=None,diccionario=None):

    n_rows_train = self.datosTrain.shape[0]
    n_rows_test = datostest.shape[0]
    n_atts = datostest.shape[1] - 1

    matrizDistancias = np.empty((n_rows_train, n_rows_test))

    # Normalizamos los datostest usando la Z-Score
    for i in range(n_atts):
      media = np.mean(datostest[:, i])
      varianza = np.std(datostest[:, i])
      datostest[:, i] = (datostest[:, i] - media) / varianza

    # Calculamos una matriz de distancias de todos los puntos del dataset de test con train
    for i in range(n_rows_train):
      for j in range(n_rows_test):
        matrizDistancias[i, j] = distancia.calcular(self.datosTrain[i, :-1], datostest[j, :-1])
    
    prediccion = []
    # Calculamos los k vecinos mas cercanos a cada punto de test
    for i in range(matrizDistancias.shape[1]):
      col = matrizDistancias[:, i]
      closestNeighbors = np.argsort(col)[:k]
      # Cogemos las clases pertenecientes a dichos vecinos
      clases = []
      for elem in closestNeighbors:
        clases.append(self.datosTrain[elem, -1])
      # Devolvemos la clase más repetida
      prediccion.append(np.bincount(clases).argmax())

    return np.array(prediccion)
  
  def reset_clasificador(self):
    self.datosTrain = None





class Distancia(object, metaclass=ABCMeta):
  @abstractmethod
  def calcular(self, p1, p2):
    pass


class DistanciaEuclidea(Distancia):

  # Calcula la distancia euclidea entre dos puntos cualesquiera 
  def calcular(self, p1, p2):
    if np.all(np.equal(p1, p2)):
      return 0

    suma = 0
    for e1, e2 in zip(p1, p2):
      suma += pow(e1 - e2, 2)
    
    return math.sqrt(suma)


class DistanciaManhattan(Distancia):

  # Calcula la distancia manhattan entre dos puntos cualesquiera 
  def calcular(self, p1, p2):
    if np.array_equal(p1, p2):
      return 0

    suma = 0
    for e1, e2 in zip(p1, p2):
      suma += abs(e1 - e2)

    return suma
  

class ClasificadorRegresionLogistica(Clasificador):
  
  # Vector de coeficientes que modificará el método de entrenamiento
  def __init__(self):
    self.coef = None

  def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None, step=1, epocas=1000):
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
    for e in range(epocas):
      for r in range(n_rows_train):
        s = sigmoid(self.coef, datostrain[r, :-1])
        self.coef -= step * (s - datostrain[r, -1]) * datostrain[r, :-1]

  
  # Clasifica en funcion d ela salida de la funcion sigmoidal
  def clasifica(self,datosTest,atributosDiscretos=None, diccionario=None):
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
  return 1 / (1 + np.exp(-a))