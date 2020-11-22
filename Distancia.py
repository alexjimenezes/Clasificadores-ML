from abc import ABCMeta,abstractmethod
import numpy as np
import math

class Distancia(object, metaclass=ABCMeta):
  @abstractmethod
  def calcular(self, p1, p2, VI=None):
    pass


class DistanciaEuclidea(Distancia):

  def __str__(self):
    return "Distancia Euclidea"

  # Calcula la distancia euclidea entre dos puntos cualesquiera 
  def calcular(self, p1, p2):
    if np.all(np.equal(p1, p2)):
      return 0

    suma = 0
    for e1, e2 in zip(p1, p2):
      suma += pow(e1 - e2, 2)
    
    return math.sqrt(suma)


class DistanciaManhattan(Distancia):

  def __str__(self):
    return "Distancia Manhattan"

  # Calcula la distancia manhattan entre dos puntos cualesquiera 
  def calcular(self, p1, p2):
    if np.array_equal(p1, p2):
      return 0

    suma = 0
    for e1, e2 in zip(p1, p2):
      suma += abs(e1 - e2)

    return suma

class DistanciaMahalanobis(Distancia):

  def __str__(self):
    return "Distancia Mahalanobis"

  # Clacula la distancia Mahalanobis entre un punto y
  def calcular(self, p1, p2, VI=None):
    # sqrt((p1-p2)' * V⁻¹ * (p1-p2))
    return math.sqrt(np.dot(np.dot(np.transpose(p1 - p2), VI), (p1 - p2)))