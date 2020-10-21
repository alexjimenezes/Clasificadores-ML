from abc import ABCMeta,abstractmethod
import numpy as np

class Clasificador(object, metaclass=ABCMeta):
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
"""   @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass 
  
  
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
    self.tablaProb = []


  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):

    # TODO: headers = datostrain[0]
    matrix = datostrain
    matrixPorClases = []
    numClases = len(set(matrix[:,-1]))
    numTypesPerAttribute = []
    self.mediaVarianza = np.zeros((matrix.shape[1]-1, numClases))

    # Separa matriz por clases
    for c in range(numClases):
      matrixPorClases.append([])
      for i in range(matrix.shape[0]):
        if matrix[i, -1] == c:
          matrixPorClases[c].append(matrix[i, :])
        

    # TODO: usar diccionarios para obtener longitus de typos
    for i in range(matrix.shape[1]-1):
      if atributosDiscretos[i]:
        numTypesPerAttribute.append(len(set(matrix[:][i])))
      else:
        for clase in range(numClases):
          self.mediaVarianza[i, c] = [np.mean(matrixPorClases[clase][:][i]), np.std(matrixPorClases[clase][:][i])]
    print(self.mediaVarianza)

    #self.mediaVarianza[i, j]=([np.mean(matrix[:,i]), np.std(matrix[:,i])])

    for k in range(len(numTypesPerAttribute)):
      self.tablaProb.append(np.zeros((numClases, numTypesPerAttribute[k])))

    # Sumar 1 cuando se encuentre un ejemplo
    for i in range(matrix.shape[1]-1):
      if atributosDiscretos[i]:
        for l in range(matrix.shape[0]):
          self.tablaProb[i][int(matrix[l, -1]),int(matrix[l,i])] += 1

    # Laplace en caso de ser necesario
    for i in range(matrix.shape[1]-1):
      if atributosDiscretos[i]:
        flag = 0
        for c in self.tablaProb[i]:
          for a in range(len(c)):
            if c[a] == 0:
              aplicar_laplace(i, a)
              flag = 1
            break
          if flag == 1:
            break

  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
    # computar por la formula  

   pass
  def aplicar_laplace(self, idxatributo, idxvalorcolumna):
    self.tablaProb[idxatributo][:,idxvalorcolumna] += 1
