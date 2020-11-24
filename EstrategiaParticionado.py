# -*- coding: utf-8 -*-

# coding: utf-8
#Comentario de prueba
from abc import ABCMeta,abstractmethod
import random

class Particion():

  # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado(object, metaclass=ABCMeta):
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor
  @abstractmethod
  def __init__(self):
    self.particiones = list()
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  
  def __init__(self, proporcionTest, numeroEjecuciones):
    super().__init__()
    self.proporcionTest = proporcionTest
    self.numeroEjecuciones = numeroEjecuciones

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
  # Devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):
    self.particiones = list()
    random.seed(seed)

    # Get how many rows will be used for test & train
    idxForTest = int(self.proporcionTest * datos.shape[0])

    # Create same number of partitions as number of iterations exist
    for i in range(self.numeroEjecuciones):
      idx = list(range(datos.shape[0]))
      random.shuffle(idx)
      p = Particion()
      p.indicesTest = idx[:idxForTest]
      p.indicesTrain = idx[idxForTest:]
      self.particiones.append(p)

    return self.particiones
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):

  def __init__(self, numeroParticiones):
    super().__init__()
    self.numeroParticiones = numeroParticiones

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):  
    self.particiones = list()
    random.seed(seed)
    resto = datos.shape[0] % self.numeroParticiones
    rows = int(datos.shape[0] / self.numeroParticiones)
    last_index = 0
    indices = list(range(datos.shape[0]))
    random.shuffle(indices)
    for i in range(self.numeroParticiones):
      numero_rows=rows
      if resto > 0:
        numero_rows += 1
        resto -= 1
      p = Particion()
      p.indicesTest = indices[last_index : last_index + numero_rows]
      p.indicesTrain = indices[:last_index] + indices[last_index + numero_rows : datos.shape[0]] 
      last_index += numero_rows
      self.particiones.append(p)
  
    return self.particiones




 
    
