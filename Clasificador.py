from abc import ABCMeta,abstractmethod
import numpy as np
from scipy.stats import norm

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

"""
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error    
	  pass
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
  	  pass  
"""

##############################################################################

class ClasificadorNaiveBayes(Clasificador):

  def __init__(self):
    self.tablaNominales = dict()
    self.tablaContinuos = dict()
    self.apriori = dict()


  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
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
        # Apply laplace where neccessary
        if np.any(self.tablaNominales[i] == 0):
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
            prob = ocurrencia_value / sum_class
            multiplicador[c] *= prob
          else:
            value = datostest[row][i]
            mean = self.tablaContinuos[i][MEDIA_V, c]
            std = self.tablaContinuos[i][DESVIACION_V, c]
            prob = norm.pdf(value, loc=mean, scale=std)
            multiplicador[c] *= prob
      predicciones[row] = max(multiplicador, key=multiplicador.get)
    
    return predicciones
